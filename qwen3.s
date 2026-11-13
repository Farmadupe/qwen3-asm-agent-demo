# qwen3.s - Qwen3-0.6B greedy decoder, x86-64 Linux, AVX2 + FMA, no libc.
#
#   stdin     prefix as little-endian u32 token IDs (read until EOF)
#   stdout    generated token IDs as little-endian u32, one write() per token
#   argv[1]   optional cap on generated tokens (default: until EOS or context full)
#   argv[2]   optional; nonzero makes the engine write, for every position it computes
#             (every prefix token, then every decoded token that is fed back), the
#             VOCAB f32 logits (little endian, 4*VOCAB bytes, no delimiter) to stderr
#   exit      0 ok, 1 usage, 2 weights missing/mismatched, 3 read error,
#             4 bad prefix, 5 write error, 6 out of memory
#
# Weights are read from a blob appended to this executable by
# tools/attach_weights.py. Weights are bf16 in the blob and are converted to
# f32 on the fly; activations, KV cache and accumulation are f32.
#
# Notes on the implementation:
#   * Prefill runs in groups of up to 4 tokens so each weight row is read once
#     per group; decode runs one token at a time.
#   * Logits are only computed for the last prefix token and for decoded tokens.
#   * Matrix-vector products (output rows) and attention (batch x head tasks) are
#     split across a thread pool: min(CPUs in affinity mask, MAXTHREADS) threads,
#     created with clone(). Workers spin briefly between jobs and then sleep on a
#     futex, so an idle engine does not burn CPU.
#   * Sampling is greedy argmax (first index wins ties).

.intel_syntax noprefix

.set H,        1024
.set NL,       28
.set NH,       16
.set NKV,      8
.set HD,       128
.set QD,       2048          # NH * HD
.set KVD,      1024          # NKV * HD
.set FF,       3072
.set VOCAB,    151936
.set MAXCTX,   8192
.set EOS_A,    151643        # <|endoftext|>
.set EOS_B,    151645        # <|im_end|>
.set NSLOTS,   312           # 2 + NL*11 + 2 (see attach_weights.py)
.set SLOT_COS, 310
.set SLOT_SIN, 311
.set KV_BYTES, 939524096     # NL * MAXCTX * KVD * 4
.set MAXTHREADS,  8
.set STACK_BYTES, 262144
.set CLONE_FLAGS, 0x50F00

# Helper macros ------------------------------------------------------------

# Load the blob pointer for weight kind k (0 ln1, 1 q, 2 k, 3 v, 4 o,
# 5 q_norm, 6 k_norm, 7 ln2, 8 gate, 9 up, 10 down) of layer g_fw_layer.
.macro LSLOT dst, k
    mov \dst, qword ptr [rip+g_fw_layer]
    imul \dst, \dst, 11
    add \dst, \k+2
    lea r11, [rip+g_slots]
    mov \dst, qword ptr [r11+\dst*8]
.endm

# Horizontal sum of ymm \y into the low lane \x (xmm), scratch xmm \t.
.macro HSUM y, x, t
    vextractf128 \t, \y, 1
    vaddps \x, \x, \t
    vmovhlps \t, \t, \x
    vaddps \x, \x, \t
    vpshufd \t, \x, 0x55
    vaddss \x, \x, \t
.endm

# 8 copies of a float, for packed memory operands.
.macro VEC name, val
    .align 32
\name:
    .rept 8
    .float \val
    .endr
.endm

# ==========================================================================
.section .text

.globl _start
_start:
    mov qword ptr [rip+g_max_new], -1
    mov rax, [rsp]                       # argc
    cmp rax, 2
    jb .Lnoarg
    mov rdi, [rsp+16]                    # argv[1]
    call parse_decimal
    test rdx, rdx
    jnz .Lusage
    mov [rip+g_max_new], rax
    cmp qword ptr [rsp], 3
    jb .Lnoarg
    mov rdi, [rsp+24]                    # argv[2]: emit logits on stderr if nonzero
    call parse_decimal
    test rdx, rdx
    jnz .Lusage
    mov [rip+g_emit], rax
.Lnoarg:

    # --- locate the weight trailer at the end of /proc/self/exe -------------
    mov eax, 2                           # open
    lea rdi, [rip+s_self]
    xor esi, esi
    xor edx, edx
    syscall
    test rax, rax
    js .Lno_weights
    mov [rip+g_fd], rax

    mov rdi, rax                         # lseek(fd, 0, SEEK_END)
    xor esi, esi
    mov edx, 2
    mov eax, 8
    syscall
    cmp rax, 24
    jb .Lno_weights
    lea rsi, [rax-24]
    mov rdi, [rip+g_fd]
    xor edx, edx                         # lseek(fd, size-24, SEEK_SET)
    mov eax, 8
    syscall

    mov rdi, [rip+g_fd]                  # read(fd, trailer, 24)
    lea rsi, [rip+g_trailer]
    mov edx, 24
    xor eax, eax
    syscall
    cmp rax, 24
    jne .Lno_weights
    mov rax, [rip+magic_trl]
    cmp rax, [rip+g_trailer+16]
    jne .Lno_weights
    mov r13, [rip+g_trailer]             # blob offset in file (page aligned)
    mov r14, [rip+g_trailer+8]           # blob size

    xor edi, edi                         # mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, off)
    mov rsi, r14
    mov edx, 1
    mov r10d, 2
    mov r8, [rip+g_fd]
    mov r9, r13
    mov eax, 9
    syscall
    cmp rax, -4096
    ja .Lmmap_fail
    mov [rip+g_blob], rax

    # --- validate blob header and build the slot table ---------------------
    mov rsi, [rip+g_blob]
    mov rax, [rip+magic_blob]
    cmp rax, [rsi]
    jne .Lbad_blob
    lea rdi, [rip+expect_cfg]
    xor ecx, ecx
.Lcfg_loop:
    mov eax, dword ptr [rsi+rcx*4+16]
    cmp eax, dword ptr [rdi+rcx*4]
    jne .Lbad_blob
    inc rcx
    cmp rcx, 8
    jb .Lcfg_loop

    lea rdi, [rip+g_slots]
    xor ecx, ecx
.Lslot_loop:
    mov rax, [rsi+rcx*8+48]
    add rax, rsi
    mov [rdi+rcx*8], rax
    inc rcx
    cmp rcx, NSLOTS
    jb .Lslot_loop

    # --- KV caches (lazily committed anonymous memory) ----------------------
    mov eax, 9                           # mmap(NULL, KV_BYTES, RW, PRIVATE|ANON|NORESERVE)
    xor edi, edi
    mov rsi, KV_BYTES
    mov edx, 3
    mov r10d, 0x4022
    mov r8, -1
    xor r9d, r9d
    syscall
    cmp rax, -4096
    ja .Lkv_fail
    mov [rip+g_kcache], rax

    mov eax, 9
    xor edi, edi
    mov rsi, KV_BYTES
    mov edx, 3
    mov r10d, 0x4022
    mov r8, -1
    xor r9d, r9d
    syscall
    cmp rax, -4096
    ja .Lkv_fail
    mov [rip+g_vcache], rax

    # --- thread pool: one thread per CPU in the affinity mask (max MAXTHREADS) ---
    mov eax, 204                         # sched_getaffinity(0, 128, mask)
    xor edi, edi
    mov esi, 128
    lea rdx, [rip+g_affinity]
    syscall
    lea r8, [rip+g_affinity]
    xor ecx, ecx
    xor eax, eax
.Lcpu_count:
    popcnt rdx, qword ptr [r8+rax*8]
    add rcx, rdx
    inc rax
    cmp rax, 16
    jb .Lcpu_count
    test rcx, rcx
    jnz .Lcpu_nonzero
    mov ecx, 1
.Lcpu_nonzero:
    cmp rcx, MAXTHREADS
    jbe .Lcpu_capped
    mov ecx, MAXTHREADS
.Lcpu_capped:
    mov [rip+g_nthreads], rcx
    mov r15, 1                           # id of the next thread to create (main is 0)
.Lspawn:
    cmp r15, [rip+g_nthreads]
    jae .Lspawn_done
    mov eax, 9                           # mmap a stack
    xor edi, edi
    mov rsi, STACK_BYTES
    mov edx, 3
    mov r10d, 0x22                       # MAP_PRIVATE | MAP_ANONYMOUS
    mov r8, -1
    xor r9d, r9d
    syscall
    cmp rax, -4096
    ja .Lspawn_fail
    lea rsi, [rax+STACK_BYTES-16]        # child rsp; the thread id sits at [rsp]
    mov [rsi], r15
    mov eax, 56                          # clone(CLONE_VM|FS|FILES|SIGHAND|THREAD|SYSVSEM)
    mov edi, CLONE_FLAGS
    xor edx, edx
    xor r10d, r10d
    xor r8d, r8d
    syscall
    test rax, rax
    jz worker_entry                      # child thread
    cmp rax, -4096
    ja .Lspawn_fail
    inc r15
    jmp .Lspawn
.Lspawn_fail:
    mov [rip+g_nthreads], r15            # keep only the threads that exist
.Lspawn_done:
    mov rax, [rip+g_nthreads]
    dec rax
    mov [rip+g_nworkers], eax

    # --- read the prefix from stdin -----------------------------------------
    xor r15d, r15d                       # bytes read so far
.Lread:
    lea rsi, [rip+g_tokbuf]
    add rsi, r15
    mov rdx, MAXCTX*4+4
    sub rdx, r15
    xor edi, edi
    xor eax, eax                         # read(0, ...)
    syscall
    test rax, rax
    js .Lread_fail
    jz .Lread_done
    add r15, rax
    cmp r15, MAXCTX*4
    ja .Lpref_long
    jmp .Lread
.Lread_done:
    test r15, 3
    jnz .Lpref_partial
    test r15, r15
    jz .Lpref_empty
    shr r15, 2                           # r15 = number of prefix tokens
    mov [rip+g_ntok], r15

    lea rsi, [rip+g_tokbuf]              # validate token IDs
    xor ecx, ecx
.Lval:
    cmp rcx, r15
    jae .Lval_ok
    mov eax, dword ptr [rsi+rcx*4]
    cmp eax, VOCAB
    jae .Lpref_range
    inc rcx
    jmp .Lval
.Lval_ok:

    # --- prefill in groups of up to 4 tokens -------------------------------
    xor r12d, r12d                       # p = first prefix token of the group
.Lpf:
    mov r13, [rip+g_ntok]
    sub r13, r12                         # remaining
    mov r14, r13
    cmp r14, 4
    jbe .Lpf_nb
    mov r14, 4
.Lpf_nb:
    lea rdi, [rip+g_tokbuf]
    lea rdi, [rdi+r12*4]
    mov rsi, r12                         # p0
    mov rdx, r14                         # nb
    lea rcx, [r12+r14]
    xor eax, eax
    cmp rcx, [rip+g_ntok]
    sete al
    mov rcx, rax                         # want logits only for the last token
    call forward
    add r12, r14
    cmp r12, [rip+g_ntok]
    jb .Lpf

    # --- decode -------------------------------------------------------------
    mov rax, [rip+g_ntok]
    mov [rip+g_pos], rax
.Ldec:
    mov rax, [rip+g_count]
    cmp rax, [rip+g_max_new]
    jae .Lfinish
    mov eax, dword ptr [rip+g_next]      # emit
    mov dword ptr [rip+g_outbuf], eax
    lea rsi, [rip+g_outbuf]
    mov edx, 4
.Lwrite:
    mov edi, 1
    mov eax, 1
    syscall
    test rax, rax
    jle .Lwrite_fail
    add rsi, rax
    sub rdx, rax
    jnz .Lwrite
    inc qword ptr [rip+g_count]

    mov eax, dword ptr [rip+g_next]
    cmp eax, EOS_A
    je .Lfinish
    cmp eax, EOS_B
    je .Lfinish
    mov rax, [rip+g_pos]
    cmp rax, MAXCTX
    jae .Lfinish                         # context full

    lea rdi, [rip+g_next]                # feed the new token at position pos
    mov rsi, rax
    mov edx, 1
    mov ecx, 1
    call forward
    inc qword ptr [rip+g_pos]
    jmp .Ldec

.Lfinish:
    xor edi, edi
    mov eax, 231                         # exit_group(0)
    syscall

# --- error exits ------------------------------------------------------------
.Lusage:
    lea rsi, [rip+msg_usage]
    mov edi, 1
    call die
.Lno_weights:
    lea rsi, [rip+msg_no_weights]
    mov edi, 2
    call die
.Lmmap_fail:
    lea rsi, [rip+msg_mmap]
    mov edi, 2
    call die
.Lbad_blob:
    lea rsi, [rip+msg_bad_blob]
    mov edi, 2
    call die
.Lkv_fail:
    lea rsi, [rip+msg_kv]
    mov edi, 6
    call die
.Lread_fail:
    lea rsi, [rip+msg_read]
    mov edi, 3
    call die
.Lpref_long:
    lea rsi, [rip+msg_long]
    mov edi, 4
    call die
.Lpref_partial:
    lea rsi, [rip+msg_partial]
    mov edi, 4
    call die
.Lpref_empty:
    lea rsi, [rip+msg_empty]
    mov edi, 4
    call die
.Lpref_range:
    lea rsi, [rip+msg_range]
    mov edi, 4
    call die
.Lwrite_fail:
    lea rsi, [rip+msg_write]
    mov edi, 5
    call die

# ==========================================================================
# parse_decimal: rdi = NUL-terminated string -> rax = value, rdx = 0 ok / 1 bad
parse_decimal:
    xor eax, eax
    xor ecx, ecx
.Lpd:
    movzx r8d, byte ptr [rdi+rcx]
    test r8d, r8d
    jz .Lpd_end
    sub r8d, 48
    cmp r8d, 9
    ja .Lpd_bad
    imul rax, rax, 10
    add rax, r8
    inc rcx
    jmp .Lpd
.Lpd_end:
    test rcx, rcx
    jz .Lpd_bad
    xor edx, edx
    ret
.Lpd_bad:
    mov edx, 1
    ret

# die: edi = exit code, rsi = NUL-terminated message (written to stderr)
die:
    mov r12d, edi
    xor edx, edx
.Lddl:
    cmp byte ptr [rsi+rdx], 0
    je .Ldw
    inc rdx
    jmp .Lddl
.Ldw:
    mov edi, 2
    mov eax, 1
    syscall
    mov edi, r12d
    mov eax, 231
    syscall

# ==========================================================================
# bf16_to_f32: rdi = dst f32, rsi = src bf16, rdx = n (multiple of 8)
bf16_to_f32:
    xor eax, eax
.Lb2f:
    vpmovzxwd ymm0, XMMWORD PTR [rsi+rax*2]
    vpslld ymm0, ymm0, 16
    vmovups YMMWORD PTR [rdi+rax*4], ymm0
    add rax, 8
    cmp rax, rdx
    jb .Lb2f
    vzeroupper
    ret

# rmsnorm: dst = src * rsqrt(mean(src^2) + eps) * w
#   rdi = dst f32, rsi = src f32 (may equal dst), rdx = w bf16, rcx = n (multiple of 8)
rmsnorm:
    mov r8, rcx
    vxorps ymm0, ymm0, ymm0
    xor eax, eax
.Lrn_ss:
    vmovups ymm1, YMMWORD PTR [rsi+rax*4]
    vfmadd231ps ymm0, ymm1, ymm1
    add rax, 8
    cmp rax, rcx
    jb .Lrn_ss
    HSUM ymm0, xmm0, xmm1
    vcvtsi2ss xmm1, xmm1, r8d
    vdivss xmm0, xmm0, xmm1              # mean of squares
    vaddss xmm0, xmm0, DWORD PTR [rip+S_EPS]
    vsqrtss xmm0, xmm0, xmm0
    vmovss xmm1, DWORD PTR [rip+S_ONE]
    vdivss xmm0, xmm1, xmm0              # 1 / rms
    vbroadcastss ymm2, xmm0
    xor eax, eax
.Lrn_sc:
    vmovups ymm1, YMMWORD PTR [rsi+rax*4]
    vmulps ymm1, ymm1, ymm2
    vpmovzxwd ymm3, XMMWORD PTR [rdx+rax*2]
    vpslld ymm3, ymm3, 16
    vmulps ymm1, ymm1, ymm3
    vmovups YMMWORD PTR [rdi+rax*4], ymm1
    add rax, 8
    cmp rax, rcx
    jb .Lrn_sc
    vzeroupper
    ret

# rope (rotate-half, as in HF Qwen3): rdi = base of heads (f32, 128 per head),
#   rsi = number of heads, rdx = position
rope:
    mov rax, rdx
    shl rax, 8                           # pos * 64 floats * 4 bytes
    lea r8, [rip+g_slots]
    mov r9, qword ptr [r8+SLOT_COS*8]
    add r9, rax
    mov r10, qword ptr [r8+SLOT_SIN*8]
    add r10, rax
    xor r11d, r11d
.Lrope_h:
    cmp r11, rsi
    jae .Lrope_done
    mov rax, r11
    shl rax, 9                           # head * 128 * 4
    lea rcx, [rdi+rax]
    xor eax, eax
.Lrope_i:
    vmovups ymm0, YMMWORD PTR [rcx+rax*4]        # x1 = q[i]
    vmovups ymm1, YMMWORD PTR [rcx+rax*4+256]    # x2 = q[i+64]
    vmovups ymm2, YMMWORD PTR [r9+rax*4]         # cos
    vmovups ymm3, YMMWORD PTR [r10+rax*4]        # sin
    vmulps ymm4, ymm0, ymm2
    vfnmadd231ps ymm4, ymm1, ymm3                # x1*cos - x2*sin
    vmulps ymm5, ymm1, ymm2
    vfmadd231ps ymm5, ymm0, ymm3                 # x2*cos + x1*sin
    vmovups YMMWORD PTR [rcx+rax*4], ymm4
    vmovups YMMWORD PTR [rcx+rax*4+256], ymm5
    add rax, 8
    cmp rax, 64
    jb .Lrope_i
    inc r11
    jmp .Lrope_h
.Lrope_done:
    vzeroupper
    ret

# matvec: dst[b][r] = sum_c W[r][c] * X[b][c], W bf16 row-major
#   rdi = dst f32 (batch stride = rows), rsi = W bf16, rdx = X f32 (batch stride = cols),
#   rcx = rows, r8 = cols (multiple of 32), r9 = nb (batch size)
# Rows are split across the thread pool (run_parallel) for every group of 1 or 4 vectors.
matvec:
    mov [rip+g_mv_dst], rdi
    mov [rip+g_mv_w], rsi
    mov [rip+g_mv_x], rdx
    mov [rip+g_mv_rows], rcx
    mov [rip+g_mv_cols], r8
    mov [rip+g_mv_nb], r9
.Lmv_loop:
    mov r9, [rip+g_mv_nb]
    test r9, r9
    jz .Lmv_ret
    mov rax, [rip+g_mv_dst]
    mov [rip+g_job_dst], rax
    mov rax, [rip+g_mv_w]
    mov [rip+g_job_w], rax
    mov rax, [rip+g_mv_x]
    mov [rip+g_job_x], rax
    mov rax, [rip+g_mv_rows]
    mov [rip+g_job_rows], rax
    mov rax, [rip+g_mv_cols]
    mov [rip+g_job_cols], rax
    mov qword ptr [rip+g_job_kind], 1
    cmp r9, 4
    jb .Lmv_one
    mov qword ptr [rip+g_job_kind], 4
.Lmv_one:
    call run_parallel
    mov rax, [rip+g_job_kind]
    cmp rax, 4
    jne .Lmv_adv1
    mov rax, [rip+g_mv_rows]
    shl rax, 4                           # 4 outputs * rows * 4 bytes
    add [rip+g_mv_dst], rax
    mov rax, [rip+g_mv_cols]
    shl rax, 4
    add [rip+g_mv_x], rax
    sub qword ptr [rip+g_mv_nb], 4
    jmp .Lmv_loop
.Lmv_adv1:
    mov rax, [rip+g_mv_rows]
    shl rax, 2
    add [rip+g_mv_dst], rax
    mov rax, [rip+g_mv_cols]
    shl rax, 2
    add [rip+g_mv_x], rax
    dec qword ptr [rip+g_mv_nb]
    jmp .Lmv_loop
.Lmv_ret:
    ret

# run_parallel: run the job in g_job_* (kind 1 or 4) on all threads and wait for all.
run_parallel:
    mov dword ptr [rip+g_done], 0
    lock inc dword ptr [rip+g_gen]       # publish the job
    lea rdi, [rip+g_gen]
    mov esi, 129                         # FUTEX_WAKE | FUTEX_PRIVATE_FLAG
    mov edx, 0x7fffffff
    mov eax, 202
    syscall
    xor edi, edi                         # thread 0 is the caller
    call run_share
    mov eax, dword ptr [rip+g_nworkers]
.Lrp_wait:
    cmp dword ptr [rip+g_done], eax
    je .Lrp_done
    pause
    jmp .Lrp_wait
.Lrp_done:
    ret

# run_share: thread rdi computes its slice of the rows of the current job.
run_share:
    push rbx
    push r12
    push r13
    mov rbx, rdi
    cmp qword ptr [rip+g_job_kind], 2
    jne .Lrs_matvec
    mov rdi, rbx
    call attn_share
    jmp .Lrs_ret
.Lrs_matvec:
    mov rax, [rip+g_job_rows]
    xor edx, edx
    mov rcx, [rip+g_nthreads]
    div rcx
    mov r12, rax                         # chunk = rows / nthreads
    mov r13, rbx
    imul r13, r12                        # first row of this slice
    mov rcx, [rip+g_job_rows]
    sub rcx, r13                         # rows left
    mov rax, [rip+g_nthreads]
    dec rax
    cmp rbx, rax
    je .Lrs_have_n                       # last thread takes the remainder
    mov rcx, r12
.Lrs_have_n:
    test rcx, rcx
    jz .Lrs_ret
    mov rdi, [rip+g_job_dst]
    lea rdi, [rdi+r13*4]
    mov rsi, [rip+g_job_cols]
    mov rax, r13
    imul rax, rsi
    mov rsi, [rip+g_job_w]
    lea rsi, [rsi+rax*2]                 # W row r0
    mov rdx, [rip+g_job_x]
    mov r8, [rip+g_job_cols]
    mov r9, [rip+g_job_rows]
    shl r9, 2                            # output batch stride in bytes
    cmp qword ptr [rip+g_job_kind], 4
    je .Lrs_k4
    call kernel1
    jmp .Lrs_ret
.Lrs_k4:
    call kernel4
.Lrs_ret:
    pop r13
    pop r12
    pop rbx
    ret

# worker threads: wait for a new job generation, run their slice, count completion.
worker_entry:
    mov rbx, [rsp]                       # thread id, stored at the stack top by _start
    xor r12d, r12d                       # last generation handled (starts at 0)
.Lw_loop:
    mov ecx, 20000000
.Lw_spin:
    mov eax, dword ptr [rip+g_gen]
    cmp eax, r12d
    jne .Lw_have
    pause
    dec ecx
    jnz .Lw_spin
    lea rdi, [rip+g_gen]                 # sleep until g_gen changes
    mov esi, 128                         # FUTEX_WAIT | FUTEX_PRIVATE_FLAG
    mov edx, r12d
    xor r10d, r10d
    mov eax, 202
    syscall
    jmp .Lw_loop
.Lw_have:
    mov r12d, eax
    mov rdi, rbx
    call run_share
    lock inc dword ptr [rip+g_done]
    jmp .Lw_loop

# kernel1: one batch vector, rows [0, rcx) of a row-block.
#   rdi = out, rsi = W (first row of block), rdx = x, rcx = rows, r8 = cols
kernel1:
    xor r10d, r10d
.Lk1_row:
    cmp r10, rcx
    jae .Lk1_done
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    vxorps ymm2, ymm2, ymm2
    vxorps ymm3, ymm3, ymm3
    xor eax, eax
.Lk1_col:
    vpmovzxwd ymm4, XMMWORD PTR [rsi+rax*2]
    vpmovzxwd ymm5, XMMWORD PTR [rsi+rax*2+16]
    vpmovzxwd ymm6, XMMWORD PTR [rsi+rax*2+32]
    vpmovzxwd ymm7, XMMWORD PTR [rsi+rax*2+48]
    vpslld ymm4, ymm4, 16
    vpslld ymm5, ymm5, 16
    vpslld ymm6, ymm6, 16
    vpslld ymm7, ymm7, 16
    vfmadd231ps ymm0, ymm4, YMMWORD PTR [rdx+rax*4]
    vfmadd231ps ymm1, ymm5, YMMWORD PTR [rdx+rax*4+32]
    vfmadd231ps ymm2, ymm6, YMMWORD PTR [rdx+rax*4+64]
    vfmadd231ps ymm3, ymm7, YMMWORD PTR [rdx+rax*4+96]
    add rax, 32
    cmp rax, r8
    jb .Lk1_col
    vaddps ymm0, ymm0, ymm1
    vaddps ymm2, ymm2, ymm3
    vaddps ymm0, ymm0, ymm2
    HSUM ymm0, xmm0, xmm4
    vmovss DWORD PTR [rdi+r10*4], xmm0
    lea rsi, [rsi+r8*2]
    inc r10
    jmp .Lk1_row
.Lk1_done:
    vzeroupper
    ret

# kernel4: four batch vectors at once, rows [0, rcx) of a row-block.
#   rdi = out for batch 0, r9 = output batch stride (bytes), rsi = W, rdx = x for batch 0,
#   x for batch i is at rdx + i*cols*4, rcx = rows, r8 = cols
kernel4:
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov rax, r8
    shl rax, 2                           # cols * 4 bytes
    lea r12, [rdx+rax]
    lea r13, [r12+rax]
    lea r14, [r13+rax]
    lea r15, [rdi+r9]
    lea rbp, [r15+r9]
    lea rbx, [rbp+r9]
    xor r10d, r10d
.Lk4_row:
    cmp r10, rcx
    jae .Lk4_done
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    vxorps ymm2, ymm2, ymm2
    vxorps ymm3, ymm3, ymm3
    xor eax, eax
.Lk4_col:
    vpmovzxwd ymm4, XMMWORD PTR [rsi+rax*2]
    vpslld ymm4, ymm4, 16
    vfmadd231ps ymm0, ymm4, YMMWORD PTR [rdx+rax*4]
    vfmadd231ps ymm1, ymm4, YMMWORD PTR [r12+rax*4]
    vfmadd231ps ymm2, ymm4, YMMWORD PTR [r13+rax*4]
    vfmadd231ps ymm3, ymm4, YMMWORD PTR [r14+rax*4]
    add rax, 8
    cmp rax, r8
    jb .Lk4_col
    HSUM ymm0, xmm0, xmm4
    HSUM ymm1, xmm1, xmm4
    HSUM ymm2, xmm2, xmm4
    HSUM ymm3, xmm3, xmm4
    vmovss DWORD PTR [rdi+r10*4], xmm0
    vmovss DWORD PTR [r15+r10*4], xmm1
    vmovss DWORD PTR [rbp+r10*4], xmm2
    vmovss DWORD PTR [rbx+r10*4], xmm3
    lea rsi, [rsi+r8*2]
    inc r10
    jmp .Lk4_row
.Lk4_done:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

# exp8: ymm0 = exp(ymm0) for 8 lanes. Clobbers ymm0-ymm3 only.
#   Range reduction x = n*ln2 + r, polynomial for exp(r) (Cephes expf), 2^n by bit construction.
exp8:
    vminps ymm0, ymm0, YMMWORD PTR [rip+V_EXPHI]
    vmaxps ymm0, ymm0, YMMWORD PTR [rip+V_EXPLO]
    vmulps ymm1, ymm0, YMMWORD PTR [rip+V_LOG2E]
    vroundps ymm1, ymm1, 0               # n = round(x / ln2), nearest
    vmovaps ymm2, ymm0
    vfnmadd231ps ymm2, ymm1, YMMWORD PTR [rip+V_LN2HI]
    vfnmadd231ps ymm2, ymm1, YMMWORD PTR [rip+V_LN2LO]     # r = x - n*ln2
    vmovups ymm3, YMMWORD PTR [rip+V_P0]
    vfmadd213ps ymm3, ymm2, YMMWORD PTR [rip+V_P1]
    vfmadd213ps ymm3, ymm2, YMMWORD PTR [rip+V_P2]
    vfmadd213ps ymm3, ymm2, YMMWORD PTR [rip+V_P3]
    vfmadd213ps ymm3, ymm2, YMMWORD PTR [rip+V_P4]
    vfmadd213ps ymm3, ymm2, YMMWORD PTR [rip+V_P5]      # p = poly(r)
    vmulps ymm0, ymm2, ymm2                              # r^2
    vfmadd213ps ymm3, ymm0, ymm2                         # p = p*r^2 + r
    vaddps ymm3, ymm3, YMMWORD PTR [rip+V_ONE]           # exp(r)
    vcvtps2dq ymm1, ymm1
    vpaddd ymm1, ymm1, YMMWORD PTR [rip+V_I127]
    vpslld ymm1, ymm1, 23                                # 2^n
    vmulps ymm0, ymm3, ymm1
    ret

# silu_mul: gate[i] = silu(gate[i]) * up[i]
#   rdi = gate f32, rsi = up f32, rdx = n (multiple of 8)
silu_mul:
    xor eax, eax
.Lsm:
    vmovups ymm5, YMMWORD PTR [rdi+rax*4]        # g
    vxorps ymm0, ymm0, ymm0
    vsubps ymm0, ymm0, ymm5                      # -g
    call exp8
    vaddps ymm0, ymm0, YMMWORD PTR [rip+V_ONE]   # 1 + exp(-g)
    vdivps ymm1, ymm5, ymm0                      # g * sigmoid(g)
    vmulps ymm1, ymm1, YMMWORD PTR [rsi+rax*4]
    vmovups YMMWORD PTR [rdi+rax*4], ymm1
    add rax, 8
    cmp rax, rdx
    jb .Lsm
    vzeroupper
    ret

# add_vec: dst[i] += src[i]. rdi = dst, rsi = src, rdx = n (multiple of 8)
add_vec:
    xor eax, eax
.Lav:
    vmovups ymm0, YMMWORD PTR [rdi+rax*4]
    vaddps ymm0, ymm0, YMMWORD PTR [rsi+rax*4]
    vmovups YMMWORD PTR [rdi+rax*4], ymm0
    add rax, 8
    cmp rax, rdx
    jb .Lav
    vzeroupper
    ret

# argmax: rdi = f32 array, rsi = n -> eax = index of first maximum
argmax:
    xor eax, eax
    vmovss xmm0, DWORD PTR [rdi]
    mov ecx, 1
.Lam:
    cmp rcx, rsi
    jae .Lam_done
    vmovss xmm1, DWORD PTR [rdi+rcx*4]
    vucomiss xmm1, xmm0
    jbe .Lam_next
    vmovaps xmm0, xmm1
    mov eax, ecx
.Lam_next:
    inc rcx
    jmp .Lam
.Lam_done:
    ret

# attn_head: one query head over T keys (GQA: caller passes the shared kv head).
#   rdi = q head (128 f32), rsi = K head base (row stride KVD*4), rdx = V head base,
#   rcx = out head (128 f32), r8 = T (>= 1), r9 = scores scratch (MAXCTX+16 f32)
attn_head:
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov r13, rdi                         # q
    mov r14, rsi                         # K head base
    mov r15, rdx                         # V head base
    mov rbp, rcx                         # out
    mov rbx, r8                          # T
    mov r12, r9                          # scores

    # scores[t] = (q . k_t) / sqrt(HD)
    xor r9d, r9d
.Lah_sc:
    cmp r9, rbx
    jae .Lah_sc_done
    mov rsi, r9
    shl rsi, 12                          # t * KVD * 4
    add rsi, r14
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    xor r10d, r10d
.Lah_dot:
    vmovups ymm2, YMMWORD PTR [r13+r10*4]
    vfmadd231ps ymm0, ymm2, YMMWORD PTR [rsi+r10*4]
    vmovups ymm3, YMMWORD PTR [r13+r10*4+32]
    vfmadd231ps ymm1, ymm3, YMMWORD PTR [rsi+r10*4+32]
    add r10, 16
    cmp r10, HD
    jb .Lah_dot
    vaddps ymm0, ymm0, ymm1
    HSUM ymm0, xmm0, xmm2
    vmulss xmm0, xmm0, DWORD PTR [rip+S_SCALE]
    vmovss DWORD PTR [r12+r9*4], xmm0
    inc r9
    jmp .Lah_sc
.Lah_sc_done:

    # pad to a multiple of 8 with a very negative value (exp -> 0)
    mov r10, rbx
    add r10, 7
    and r10, -8                          # Tpad
    mov r9, rbx
    vmovss xmm3, DWORD PTR [rip+S_NEG]
.Lah_pad:
    cmp r9, r10
    jae .Lah_pad_done
    vmovss DWORD PTR [r12+r9*4], xmm3
    inc r9
    jmp .Lah_pad
.Lah_pad_done:

    # softmax
    vmovss xmm1, DWORD PTR [r12]
    mov r9, 1
.Lah_max:
    cmp r9, rbx
    jae .Lah_max_done
    vmaxss xmm1, xmm1, DWORD PTR [r12+r9*4]
    inc r9
    jmp .Lah_max
.Lah_max_done:
    vbroadcastss ymm7, xmm1
    vxorps ymm8, ymm8, ymm8
    xor r9d, r9d
.Lah_exp:
    cmp r9, r10
    jae .Lah_exp_done
    vmovups ymm0, YMMWORD PTR [r12+r9*4]
    vsubps ymm0, ymm0, ymm7
    call exp8
    vmovups YMMWORD PTR [r12+r9*4], ymm0
    vaddps ymm8, ymm8, ymm0
    add r9, 8
    jmp .Lah_exp
.Lah_exp_done:
    HSUM ymm8, xmm8, xmm2
    vmovss xmm2, DWORD PTR [rip+S_ONE]
    vdivss xmm8, xmm2, xmm8
    vbroadcastss ymm9, xmm8
    xor r9d, r9d
.Lah_norm:
    cmp r9, r10
    jae .Lah_norm_done
    vmovups ymm0, YMMWORD PTR [r12+r9*4]
    vmulps ymm0, ymm0, ymm9
    vmovups YMMWORD PTR [r12+r9*4], ymm0
    add r9, 8
    jmp .Lah_norm
.Lah_norm_done:

    # out = sum_t p[t] * v_t
    vxorps ymm0, ymm0, ymm0
    xor eax, eax
.Lah_zero:
    vmovups YMMWORD PTR [rbp+rax*4], ymm0
    add rax, 8
    cmp rax, HD
    jb .Lah_zero
    xor r9d, r9d
.Lah_acc:
    cmp r9, rbx
    jae .Lah_done
    vbroadcastss ymm1, DWORD PTR [r12+r9*4]
    mov rsi, r9
    shl rsi, 12
    add rsi, r15                         # V row t
    xor r10d, r10d
.Lah_acc_d:
    vmovups ymm2, YMMWORD PTR [rbp+r10*4]
    vfmadd231ps ymm2, ymm1, YMMWORD PTR [rsi+r10*4]
    vmovups YMMWORD PTR [rbp+r10*4], ymm2
    add r10, 8
    cmp r10, HD
    jb .Lah_acc_d
    inc r9
    jmp .Lah_acc
.Lah_done:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

# attn_share: thread rdi takes the (batch, head) tasks i with i % nthreads == thread.
#   Uses g_job_kbase, g_job_vbase (this layer's caches), g_job_p0, g_job_nb.
attn_share:
    push rbx
    push r12
    push r13
    push r14
    mov rbx, rdi
    mov r14, [rip+g_nthreads]
    mov r13, [rip+g_job_nb]
    shl r13, 4                           # nb * NH tasks
    mov r12, rbx
.Lash_loop:
    cmp r12, r13
    jae .Lash_done
    mov r8, r12
    shr r8, 4                            # b
    mov r9, r12
    and r9, 15                           # h
    mov rax, r8
    shl rax, 13                          # b * QD * 4
    mov r10, r9
    shl r10, 9                           # h * HD * 4
    add r10, rax                         # byte offset of this head in q / att
    lea rdi, [rip+g_q]
    add rdi, r10
    lea rcx, [rip+g_att]
    add rcx, r10
    mov rax, r9
    shr rax, 1
    shl rax, 9                           # kv head * HD * 4
    mov rsi, [rip+g_job_kbase]
    add rsi, rax
    mov rdx, [rip+g_job_vbase]
    add rdx, rax
    mov r8, r12
    shr r8, 4
    add r8, [rip+g_job_p0]
    inc r8                               # T = p0 + b + 1
    lea r9, [rip+g_scores_all]
    mov rax, rbx
    imul rax, rax, (MAXCTX+16)*4
    add r9, rax                          # this thread's scratch
    call attn_head
    add r12, r14
    jmp .Lash_loop
.Lash_done:
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

# ==========================================================================
# forward: process nb tokens (1..4) at positions p0 .. p0+nb-1.
#   rdi = pointer to nb u32 token IDs, rsi = p0, rdx = nb, rcx = want logits (0/1)
#   If want != 0, the argmax of the last token's logits is stored in g_next.
forward:
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov [rip+g_fw_tok], rdi
    mov [rip+g_fw_p0], rsi
    mov [rip+g_fw_nb], rdx
    mov [rip+g_fw_want], rcx

    # embedding: x[b] = embed[token[b]]
    xor ebx, ebx
.Lfw_emb:
    cmp rbx, [rip+g_fw_nb]
    jae .Lfw_emb_done
    mov rax, [rip+g_fw_tok]
    mov eax, dword ptr [rax+rbx*4]
    imul rax, rax, H*2
    mov rsi, qword ptr [rip+g_slots]
    add rsi, rax
    mov rdi, rbx
    shl rdi, 12
    lea rax, [rip+g_x]
    add rdi, rax
    mov edx, H
    call bf16_to_f32
    inc rbx
    jmp .Lfw_emb
.Lfw_emb_done:
    mov qword ptr [rip+g_fw_layer], 0

.Lfw_layer:
    # ---- attention block ----------------------------------------------------
    xor ebx, ebx
.Lfw_ln1:
    cmp rbx, [rip+g_fw_nb]
    jae .Lfw_ln1_done
    mov rax, rbx
    shl rax, 12
    lea rdi, [rip+g_xn]
    add rdi, rax
    lea rsi, [rip+g_x]
    add rsi, rax
    LSLOT rdx, 0
    mov ecx, H
    call rmsnorm
    inc rbx
    jmp .Lfw_ln1
.Lfw_ln1_done:

    lea rdi, [rip+g_q]
    LSLOT rsi, 1
    lea rdx, [rip+g_xn]
    mov ecx, QD
    mov r8d, H
    mov r9, [rip+g_fw_nb]
    call matvec

    lea rdi, [rip+g_k]
    LSLOT rsi, 2
    lea rdx, [rip+g_xn]
    mov ecx, KVD
    mov r8d, H
    mov r9, [rip+g_fw_nb]
    call matvec

    lea rdi, [rip+g_v]
    LSLOT rsi, 3
    lea rdx, [rip+g_xn]
    mov ecx, KVD
    mov r8d, H
    mov r9, [rip+g_fw_nb]
    call matvec

    # per-head QK RMSNorm, RoPE
    xor ebx, ebx
.Lfw_qk:
    cmp rbx, [rip+g_fw_nb]
    jae .Lfw_qk_done
    mov rax, rbx
    shl rax, 13                          # b * QD * 4
    lea r12, [rip+g_q]
    add r12, rax
    xor r13d, r13d
.Lfw_qn:
    cmp r13, NH
    jae .Lfw_qn_done
    mov rax, r13
    shl rax, 9
    lea rdi, [r12+rax]
    mov rsi, rdi
    LSLOT rdx, 5
    mov ecx, HD
    call rmsnorm
    inc r13
    jmp .Lfw_qn
.Lfw_qn_done:
    mov rax, rbx
    shl rax, 12                          # b * KVD * 4
    lea r12, [rip+g_k]
    add r12, rax
    xor r13d, r13d
.Lfw_kn:
    cmp r13, NKV
    jae .Lfw_kn_done
    mov rax, r13
    shl rax, 9
    lea rdi, [r12+rax]
    mov rsi, rdi
    LSLOT rdx, 6
    mov ecx, HD
    call rmsnorm
    inc r13
    jmp .Lfw_kn
.Lfw_kn_done:

    mov rax, rbx
    shl rax, 13
    lea rdi, [rip+g_q]
    add rdi, rax
    mov esi, NH
    mov rdx, [rip+g_fw_p0]
    add rdx, rbx
    call rope

    mov rax, rbx
    shl rax, 12
    lea rdi, [rip+g_k]
    add rdi, rax
    mov esi, NKV
    mov rdx, [rip+g_fw_p0]
    add rdx, rbx
    call rope
    inc rbx
    jmp .Lfw_qk
.Lfw_qk_done:

    # write K and V for every batch position into the cache
    xor ebx, ebx
.Lfw_kv:
    cmp rbx, [rip+g_fw_nb]
    jae .Lfw_kv_done
    mov rax, [rip+g_fw_p0]
    add rax, rbx                         # position
    mov rdx, [rip+g_fw_layer]
    imul rdx, rdx, MAXCTX
    add rdx, rax
    shl rdx, 12                          # (layer*MAXCTX + pos) * KVD * 4
    mov rax, rbx
    shl rax, 12                          # b * KVD * 4
    mov rdi, [rip+g_kcache]
    add rdi, rdx
    lea rsi, [rip+g_k]
    add rsi, rax
    mov ecx, KVD/2
    rep movsq
    mov rdi, [rip+g_vcache]
    add rdi, rdx
    lea rsi, [rip+g_v]
    add rsi, rax
    mov ecx, KVD/2
    rep movsq
    inc rbx
    jmp .Lfw_kv
.Lfw_kv_done:

    # attention for all batch positions and heads (parallel over tasks)
    mov rdx, [rip+g_fw_layer]
    imul rdx, rdx, MAXCTX*4096           # layer * MAXCTX * KVD * 4
    mov rax, [rip+g_kcache]
    add rax, rdx
    mov [rip+g_job_kbase], rax
    mov rax, [rip+g_vcache]
    add rax, rdx
    mov [rip+g_job_vbase], rax
    mov rax, [rip+g_fw_nb]
    mov [rip+g_job_nb], rax
    mov rax, [rip+g_fw_p0]
    mov [rip+g_job_p0], rax
    mov qword ptr [rip+g_job_kind], 2
    call run_parallel
.Lfw_att_done:

    # output projection and residual
    lea rdi, [rip+g_tmp]
    LSLOT rsi, 4
    lea rdx, [rip+g_att]
    mov ecx, H
    mov r8d, QD
    mov r9, [rip+g_fw_nb]
    call matvec

    xor ebx, ebx
.Lfw_res1:
    cmp rbx, [rip+g_fw_nb]
    jae .Lfw_res1_done
    mov rax, rbx
    shl rax, 12
    lea rdi, [rip+g_x]
    add rdi, rax
    lea rsi, [rip+g_tmp]
    add rsi, rax
    mov edx, H
    call add_vec
    inc rbx
    jmp .Lfw_res1
.Lfw_res1_done:

    # MLP: xn = rmsnorm(x) * ln2; h = silu(gate(xn)) * up(xn); x += down(h)
    xor ebx, ebx
.Lfw_ln2:
    cmp rbx, [rip+g_fw_nb]
    jae .Lfw_ln2_done
    mov rax, rbx
    shl rax, 12
    lea rdi, [rip+g_xn]
    add rdi, rax
    lea rsi, [rip+g_x]
    add rsi, rax
    LSLOT rdx, 7
    mov ecx, H
    call rmsnorm
    inc rbx
    jmp .Lfw_ln2
.Lfw_ln2_done:

    lea rdi, [rip+g_gate]
    LSLOT rsi, 8
    lea rdx, [rip+g_xn]
    mov ecx, FF
    mov r8d, H
    mov r9, [rip+g_fw_nb]
    call matvec

    lea rdi, [rip+g_up]
    LSLOT rsi, 9
    lea rdx, [rip+g_xn]
    mov ecx, FF
    mov r8d, H
    mov r9, [rip+g_fw_nb]
    call matvec

    xor ebx, ebx
.Lfw_silu:
    cmp rbx, [rip+g_fw_nb]
    jae .Lfw_silu_done
    mov rax, rbx
    imul rax, rax, FF*4                  # b * FF * 4
    lea rdi, [rip+g_gate]
    add rdi, rax
    lea rsi, [rip+g_up]
    add rsi, rax
    mov edx, FF
    call silu_mul
    inc rbx
    jmp .Lfw_silu
.Lfw_silu_done:

    lea rdi, [rip+g_tmp]
    LSLOT rsi, 10
    lea rdx, [rip+g_gate]
    mov ecx, H
    mov r8d, FF
    mov r9, [rip+g_fw_nb]
    call matvec

    xor ebx, ebx
.Lfw_res2:
    cmp rbx, [rip+g_fw_nb]
    jae .Lfw_res2_done
    mov rax, rbx
    shl rax, 12
    lea rdi, [rip+g_x]
    add rdi, rax
    lea rsi, [rip+g_tmp]
    add rsi, rax
    mov edx, H
    call add_vec
    inc rbx
    jmp .Lfw_res2
.Lfw_res2_done:

    inc qword ptr [rip+g_fw_layer]
    cmp qword ptr [rip+g_fw_layer], NL
    jb .Lfw_layer

    # ---- final norm + tied LM head (logits = embed . xn) --------------------
    # Normally only for the last token (want); with g_emit, for every token,
    # each one's logits written to stderr.
    mov rbx, [rip+g_fw_nb]
    dec rbx                              # b = nb-1: only the last token
    cmp qword ptr [rip+g_emit], 0
    je .Lfw_head_want
    xor ebx, ebx                         # b = 0: every token
    jmp .Lfw_head
.Lfw_head_want:
    cmp qword ptr [rip+g_fw_want], 0
    je .Lfw_ret
.Lfw_head:
    mov rax, rbx
    shl rax, 12
    lea rsi, [rip+g_x]
    add rsi, rax
    lea rdi, [rip+g_xn]
    mov rdx, qword ptr [rip+g_slots+8]   # model.norm.weight
    mov ecx, H
    call rmsnorm

    lea rdi, [rip+g_logits]
    mov rsi, qword ptr [rip+g_slots]     # embed (V x H), used as lm_head
    lea rdx, [rip+g_xn]
    mov ecx, VOCAB
    mov r8d, H
    mov r9d, 1
    call matvec

    cmp qword ptr [rip+g_emit], 0
    je .Lfw_head_next
    lea rsi, [rip+g_logits]              # write(2, logits, VOCAB*4)
    mov edx, VOCAB*4
.Lfw_elog:
    mov edi, 2
    mov eax, 1
    syscall
    test rax, rax
    jle .Lfw_elog_fail
    add rsi, rax
    sub rdx, rax
    jnz .Lfw_elog
.Lfw_head_next:
    inc rbx
    cmp rbx, [rip+g_fw_nb]
    jb .Lfw_head
    cmp qword ptr [rip+g_fw_want], 0
    je .Lfw_ret

    lea rdi, [rip+g_logits]              # g_logits holds the last token's logits
    mov esi, VOCAB
    call argmax
    mov [rip+g_next], eax

.Lfw_ret:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret
.Lfw_elog_fail:
    mov edi, 5                           # stderr is gone: exit quietly
    mov eax, 231
    syscall

# ==========================================================================
.section .rodata
s_self:      .asciz "/proc/self/exe"
magic_blob:  .ascii "QWN3BLOB"
magic_trl:   .ascii "QWN3TRL1"
.align 4
expect_cfg:  .long VOCAB, H, NL, NH, NKV, HD, FF, MAXCTX

msg_usage:      .asciz "usage: qwen3 [max_new_tokens [emit_logits]] < prefix.u32 > tokens.u32\n"
msg_no_weights: .asciz "error: no weight blob attached to this executable (run build.sh)\n"
msg_mmap:       .asciz "error: mmap of weight blob failed\n"
msg_bad_blob:   .asciz "error: weight blob does not match this engine's configuration\n"
msg_kv:         .asciz "error: could not allocate KV cache\n"
msg_read:       .asciz "error: read error on stdin\n"
msg_long:       .asciz "error: prefix longer than the 8192-token context\n"
msg_partial:    .asciz "error: stdin length is not a multiple of 4 bytes\n"
msg_empty:      .asciz "error: empty prefix\n"
msg_range:      .asciz "error: token id out of vocabulary range\n"
msg_write:      .asciz "error: write error on stdout\n"

.align 32
VEC V_EXPHI,  88.3762626647949
VEC V_EXPLO,  -87.3365
VEC V_LOG2E,  1.44269504088896341
VEC V_LN2HI,  0.693359375
VEC V_LN2LO,  -2.12194440e-4
VEC V_P0,     1.9875691500e-4
VEC V_P1,     1.3981999507e-3
VEC V_P2,     8.3334519073e-3
VEC V_P3,     4.1665795894e-2
VEC V_P4,     1.6666665459e-1
VEC V_P5,     5.0000001201e-1
VEC V_ONE,    1.0
.align 32
V_I127:
    .rept 8
    .long 127
    .endr
S_ONE:   .float 1.0
S_EPS:   .float 0.000001
S_SCALE: .float 0.08838834764831845      # 1/sqrt(128)
S_NEG:   .float -1.0e30

# ==========================================================================
.section .bss
.align 64
g_slots:    .skip NSLOTS*8               # blob pointers, one per tensor slot
g_trailer:  .skip 32
g_fd:       .skip 8
g_blob:     .skip 8
g_kcache:   .skip 8
g_vcache:   .skip 8
g_max_new:  .skip 8
g_count:    .skip 8
g_pos:      .skip 8
g_ntok:     .skip 8
g_next:     .skip 8
g_outbuf:   .skip 8
g_emit:     .skip 8
g_fw_tok:   .skip 8
g_fw_p0:    .skip 8
g_fw_nb:    .skip 8
g_fw_want:  .skip 8
g_fw_layer: .skip 8
g_mv_dst:   .skip 8
g_mv_w:     .skip 8
g_mv_x:     .skip 8
g_mv_rows:  .skip 8
g_mv_cols:  .skip 8
g_mv_nb:    .skip 8
g_gen:      .skip 8
g_done:     .skip 8
g_nthreads: .skip 8
g_nworkers: .skip 8
g_job_dst:  .skip 8
g_job_w:    .skip 8
g_job_x:    .skip 8
g_job_rows: .skip 8
g_job_cols: .skip 8
g_job_kind: .skip 8
g_job_kbase: .skip 8
g_job_vbase: .skip 8
g_job_p0:    .skip 8
g_job_nb:   .skip 8
g_affinity: .skip 128
.align 64
g_x:        .skip 4*H*4
g_xn:       .skip 4*H*4
g_tmp:      .skip 4*H*4
g_q:        .skip 4*QD*4
g_att:      .skip 4*QD*4
g_k:        .skip 4*KVD*4
g_v:        .skip 4*KVD*4
g_gate:     .skip 4*FF*4
g_up:       .skip 4*FF*4
g_logits:   .skip VOCAB*4
g_scores_all: .skip MAXTHREADS*(MAXCTX+16)*4
g_tokbuf:   .skip MAXCTX*4+16

.section .note.GNU-stack,"",@progbits
