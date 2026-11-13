# A qwen3-0.6b base inference engine written by Haiku 5.5. API cost $1.20

A Qwen3-0.6B text generator written in x86-64 assembly, with weights attached to the engine binary. To run it, execute `./inference.py "Your prefix here"`, which downloads the weights and builds the engine on first use, then streams the continuation.

Requires UV.