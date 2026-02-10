from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path=".",
    repo_id="RishiBond/engine-predictive-maintenance-final",
    repo_type="space"
)
