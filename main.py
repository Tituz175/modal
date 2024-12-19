import modal
import io
from fastapi import Response, HTTPException, Query, Request
from datetime import datetime, timezone
import requests
import os


def download_model():
    from diffusers import FluxPipeline
    import torch

    FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", 
            torch_dtype=torch.bfloat16)

image = (modal.Image.debian_slim()
        .pip_install("fastapi[standard]", "diffusers", "transformers", "accelerate", "requests", "sentencepiece")
        .run_function(download_model))


app = modal.App("sd-demo_1", image=image)


@app.cls(image=image, gpu="A100", container_idle_timeout=300, secrets=[modal.Secret.from_name("custom-secret")])
class Model:
    @modal.build()
    @modal.enter()
    def load_weights(self):
        from diffusers import FluxPipeline
        import torch

        self.pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell", 
                torch_dtype=torch.bfloat16
                )
        
        self.pipe.enable_model_cpu_offload()
        self.API_KEY = os.environ["API_KEY"]
        
# "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
    @modal.web_endpoint()
    def generate(self, request: Request, prompt: str = Query(..., description="The prompt for image generation")):
        import torch
        
        api_key = request.headers.get("SERVER-API-Key")
        if api_key != self.API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized"
                )
        
        image = self.pipe(prompt,
                          guidance_scale=0.0,
                          num_inference_steps=4,
                          max_sequence_length=256,
                          generator=torch.Generator("cpu").manual_seed(0)
                          ).images[0]
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return Response(content=buffer.getvalue(), media_type="image/jpeg")
    
    @modal.web_endpoint()
    def health(self):
        return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


# Warn-keeping function that runs every 5 minutes
@app.function(
    schedule=modal.Cron("*/5 * * * *"),
    secrets=[modal.Secret.from_name("custom-secret")],
)
def keep_warm():
    health_url = "https://tituz175--sd-demo-1-model-health.modal.run"
    generate_url = "https://tituz175--sd-demo-1-model-generate.modal.run"

    # First cheak health endpoint (no API key needed)
    health_responsse = requests.get(health_url)
    print(f"Health check at: {health_responsse.json()['timestamp']}")

    # Then make a test request to generate endpoint with API key
    headers = {"X-API-KEY": os.environ["API_KEY"]}
    generate_response = requests.get(generate_url, headers=headers)
    print(f"Generate endpoint tested successfully at: {datetime.now(timezone.utc).isoformat()}")

