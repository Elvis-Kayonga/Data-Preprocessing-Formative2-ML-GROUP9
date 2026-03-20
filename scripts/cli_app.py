import time
from voice_preprocessing import process_audio
from image_preprocessing import extract_image_features  
import numpy as np
import pandas as pd
import joblib
import logging
import warnings
import os
import sys

#Suppress ALL TensorFlow/oneDNN C++ stderr logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_VERBOSITY'] = 'ERROR'

# Redirect stderr to devnull during the TF import window
_real_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

#All TF-touching imports here, inside the silenced window 

#Restore stderr now that TF is done loading
sys.stderr.close()
sys.stderr = _real_stderr

#Safe imports

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.prompt import Prompt
    from rich.rule import Rule
    from rich.table import Table
    from rich import box
    from rich.align import Align
except ImportError:
    os.system(f"{sys.executable} -m pip install rich -q")
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.prompt import Prompt
    from rich.rule import Rule
    from rich.table import Table
    from rich import box
    from rich.align import Align

console = Console()

# Label mapping
label_to_customer_id = {
    "sharif":   128,
    "paulette": 103,
    "samuel":   152,
    "kayonga":  121,
}

# UI Helpers


def step_header(number: int, title: str):
    console.print()
    console.print(Rule(
        f"[bold cyan]Step {number}[/bold cyan]  [white]{title}[/white]",
        style="dim cyan"
    ))


def success(msg: str):
    console.print(f"\n  [bold green]✔[/bold green]  {msg}")


def warn(msg: str):
    """Non-fatal warning — prints and returns so the loop can continue."""
    console.print()
    console.print(Panel(
        f"[bold red]✘  Access Denied[/bold red]\n\n[dim]{msg}[/dim]",
        border_style="red",
        padding=(1, 4),
    ))
    console.print()


def error(msg: str):
    console.print(f"\n  [bold red]✘[/bold red]  [red]{msg}[/red]\n")


def spinner(label: str, duration: float = 1.6):
    with Progress(
        SpinnerColumn(spinner_name="dots2", style="cyan"),
        TextColumn(f"[dim]{label}[/dim]"),
        transient=True,
        console=console,
    ) as p:
        p.add_task("", total=None)
        time.sleep(duration)


def session_divider():
    console.print()
    console.print(Rule(style="dim"))
    console.print()


def load_models():
    with Progress(
        SpinnerColumn(spinner_name="dots", style="cyan"),
        TextColumn("[dim]{task.description}[/dim]"),
        BarColumn(bar_width=30, style="cyan", complete_style="green"),
        TextColumn("[green]{task.percentage:>3.0f}%[/green]"),
        transient=True,
        console=console,
    ) as p:
        task = p.add_task("Loading models…", total=6)

        face_model = joblib.load("../models/face_recognition_model.pkl")
        p.advance(task)
        voice_model = joblib.load("../models/speaker_model.pkl")
        p.advance(task)
        product_model = joblib.load("../models/product_xgb_model.pkl")
        p.advance(task)
        face_encoder = joblib.load("../encoders/face_label_encoder.pkl")
        p.advance(task)
        product_encoder = joblib.load("../encoders/product_label_encoder.pkl")
        p.advance(task)
        product_columns = joblib.load("../encoders/model_columns.pkl")
        p.advance(task)

    return face_model, voice_model, product_model, face_encoder, product_encoder, product_columns

# Single authentication attempt


def run_session(face_model, voice_model, product_model,
                face_encoder, product_encoder, product_columns,
                merged_dataset):
    """
    Runs one full authentication attempt.
    Returns True if the session completed (success or denied),
    Returns False if the user chose to cancel mid-session.
    """

    # STEP 1 — FACE RECOGNITION 
    step_header(1, "Face Recognition")

    image_path = Prompt.ask(
        "\n  [cyan]Path to your face image[/cyan]  [dim](or 'q' to quit)[/dim]"
    ).strip()

    if image_path.lower() == 'q':
        return False

    if not os.path.exists(image_path):
        error(f"File not found: {image_path}")
        return True

    spinner("Analysing facial features…", duration=1.8)

    image_features = extract_image_features(image_path)
    if image_features is None:
        error("Could not process the image. Check the file is a valid JPG/PNG.")
        return True

    predicted_face_encoded = face_model.predict(
        image_features.reshape(1, -1))[0]
    predicted_face = face_encoder.inverse_transform(
        [predicted_face_encoded])[0]

    if predicted_face.lower() == "unknown":
        warn("Your face was not recognised in our system.")
        return True

    success(
        f"Identity confirmed  →  [bold white]{predicted_face.title()}[/bold white]")

    # STEP 2 — PRODUCT PREDICTION
    step_header(2, "Product Prediction")

    customer_id = label_to_customer_id.get(predicted_face.lower())
    if customer_id is None:
        warn("Recognised face is not linked to a customer account.")
        return True

    customer_row = merged_dataset[merged_dataset['customer_id'] == int(
        customer_id)]
    if customer_row.empty:
        error("No transaction data found for this customer.")
        return True

    spinner("Computing personalised recommendation…", duration=1.4)

    product_features = customer_row.reindex(
        columns=product_columns, fill_value=0
    ).values.astype(float)
    product_encoded = product_model.predict(product_features)[0]
    product_name = product_encoder.inverse_transform([product_encoded])[0]

    success("Recommendation ready  →  [dim]pending voice verification[/dim]")

    #STEP 3 — VOICE VERIFICATION
    step_header(3, "Voice Verification")

    audio_path = Prompt.ask(
        "\n  [cyan]Path to your voice recording[/cyan]  [dim](or 'q' to quit)[/dim]"
    ).strip()

    if audio_path.lower() == 'q':
        return False

    if not os.path.exists(audio_path):
        error(f"File not found: {audio_path}")
        return True

    spinner("Analysing voiceprint…", duration=1.6)

    audio_features = process_audio(audio_path)
    if audio_features is None:
        error("Could not process the audio. Check the file is a valid WAV.")
        return True

    predicted_voice = voice_model.predict(audio_features.reshape(1, -1))[0]

    if predicted_voice.lower() != predicted_face.lower():
        warn(
            f"Voice matched '[bold]{predicted_voice.title()}[/bold]' "
            f"but face matched '[bold]{predicted_face.title()}[/bold]'. "
            f"Both identities must agree."
        )
        return True

    success(
        f"Voice confirmed  →  [bold white]{predicted_face.title()}[/bold white]")

    # RESULT 
    console.print()
    console.print(Rule(style="green"))

    result = Table(box=box.SIMPLE_HEAVY, show_header=False, padding=(0, 3))
    result.add_column(style="dim green", no_wrap=True)
    result.add_column(style="bold white")
    result.add_row("Verified user",       predicted_face.title())
    result.add_row("Recommended product", product_name)
    result.add_row("Customer ID",         str(customer_id))

    console.print(Panel(
        result,
        title="[bold green]✔  Authentication Successful[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    return True

# Main loop


def main():

    # Banner
    console.print()
    console.print(Panel.fit(
        Align.center(
            "[bold white]Secure Product Recommendation System[/bold white]\n"
            "[dim]Multimodal Authentication  ·  ML Group 9[/dim]"
        ),
        border_style="cyan",
        padding=(1, 6),
    ))
    console.print()

    # Load models once — reused across all sessions
    console.print("[dim]  Initialising system…[/dim]")
    (face_model, voice_model, product_model,
     face_encoder, product_encoder, product_columns) = load_models()
    merged_dataset = pd.read_csv("../extracted_datasets/merged_dataset.csv")
    success("All models loaded successfully")

    # Pipeline overview
    console.print()
    overview = Table(box=box.SIMPLE_HEAVY, show_header=False, padding=(0, 2))
    overview.add_column(style="dim cyan", no_wrap=True)
    overview.add_column(style="white")
    overview.add_row("Step 1", "Face recognition   — verify your identity")
    overview.add_row(
        "Step 2", "Product prediction — personalised recommendation")
    overview.add_row("Step 3", "Voice verification — confirm your identity")
    console.print(overview)
    console.print()
    console.print(
        "[dim]  Type [/dim][cyan]q[/cyan][dim] at any prompt to quit.[/dim]")

    #Session loop
    session_count = 0
    while True:
        session_count += 1

        if session_count > 1:
            session_divider()
            console.print(
                f"  [dim]New session  ·  attempt {session_count}[/dim]"
            )

        try:
            keep_running = run_session(
                face_model, voice_model, product_model,
                face_encoder, product_encoder, product_columns,
                merged_dataset,
            )
        except KeyboardInterrupt:
            keep_running = False

        if not keep_running:
            console.print()
            console.print(Panel.fit(
                Align.center("[dim]Session ended. Goodbye.[/dim]"),
                border_style="dim",
                padding=(0, 4),
            ))
            console.print()
            break

        # Prompt to run another session
        console.print()
        again = Prompt.ask(
            "  [cyan]Run another authentication?[/cyan]  [dim](y / n)[/dim]",
            choices=["y", "n", "Y", "N"],
            default="y",
            show_choices=False,
        ).lower()

        if again != "y":
            console.print()
            console.print(Panel.fit(
                Align.center("[dim]Session ended. Goodbye.[/dim]"),
                border_style="dim",
                padding=(0, 4),
            ))
            console.print()
            break


if __name__ == "__main__":
    main()
