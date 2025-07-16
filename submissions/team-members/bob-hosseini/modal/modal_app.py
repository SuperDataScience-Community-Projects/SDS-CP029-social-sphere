import shlex
import subprocess
from pathlib import Path

import modal

parent_dir = Path(__file__).parent
project_root = parent_dir.parent
streamlit_script_local_path = parent_dir / "modal_streamlit.py"
streamlit_script_remote_path = "/root/modal_streamlit.py"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "streamlit>=1.46.0",
        "pandas>=2.3.0",
        "numpy>=2.0.0",
        "matplotlib>=3.8.0",
        "plotly>=5.17.0",
        "seaborn>=0.12.0",
        "statsmodels>=0.14.0",
        "scikit-learn>=1.7.0",
        "Pillow>=10.0.0",
        "mlflow==2.22.0",
        "shap==0.48.0",
        "catboost==1.2.8",
        "xgboost==3.0.2",
        "PyYAML==6.0.2",
        )
    .add_local_file(
        streamlit_script_local_path,
        streamlit_script_remote_path,
    )
    .add_local_dir(project_root / "configs", "/root/configs")
    .add_local_dir(project_root / "src", "/root/src")
    .add_local_file(project_root / "data/data_cleaned.pickle", "/root/data/data_cleaned.pickle")
    .add_local_dir(project_root / ".streamlit", "/root/.streamlit")
    .add_local_file(project_root / "image.jpg", "/root/image.jpg")
)

app = modal.App(name="social-sphere-analytics", image=image)

if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "modal_streamlit.py not found! Place the script with your streamlit app in the same directory."
    )

@app.function()
@modal.concurrent(max_inputs=100)
@modal.web_server(8000)
def run():
    target = shlex.quote(streamlit_script_remote_path)
    cmd = f"""streamlit run {target} \
        --server.port 8000 \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false \
        --server.headless=true"""
    subprocess.Popen(cmd, shell=True)

