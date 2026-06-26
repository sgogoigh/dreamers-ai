"""Dreamers-AI trailer-generation pipeline.

Flow:  movie concept
        -> (step 1) draft scene(s) from the fine-tuned LoRA adapter   [GPU]
        -> (step 2) refine into a structured TRAILER SCRIPT           [gemini-3.5-flash]
        -> (step 3) expand each beat into a cinematic Veo 3.1 prompt  [gemini-3.5-flash]
        -> (step 4) generate & chain 8s clips into a ~1 min trailer   [veo-3.1]
"""
