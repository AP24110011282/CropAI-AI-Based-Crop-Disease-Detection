# CropAI -- AI-Based Crop Disease Detection System

CropAI is a Flask-based web application that uses a trained
deep-learning image-classification model to provide an educational
prediction of possible crop leaf diseases from uploaded images.

> **Important:** CropAI is an educational decision-support prototype. A
> model prediction is not a confirmed diagnosis and must not be used as
> the sole basis for pesticide, fertilizer, or other treatment
> decisions. Confirm suspected disease with a qualified agricultural
> expert or local agricultural extension service.

## Features

-   User registration and email/password login.
-   Authenticated image-upload and prediction workflow.
-   Image validation for JPG, JPEG, and PNG files.
-   MobileNetV2-based image classification.
-   Top-three predicted classes with confidence scores.
-   Educational disease information, including symptoms, prevention,
    affected parts, favorable conditions, monitoring, and treatment
    notes.
-   Prediction history stored in SQLite.
-   Administrator dashboard with prediction statistics and history.
-   Administrator-only prediction details, record deletion, and CSV
    export.
-   CSRF protection, request rate limiting, upload-size limits, and
    security response headers.

## Supported model classes

The supplied class-name file contains 15 classes:

1.  Bell pepper bacterial spot
2.  Bell pepper healthy
3.  Potato early blight
4.  Potato late blight
5.  Potato healthy
6.  Tomato bacterial spot
7.  Tomato early blight
8.  Tomato late blight
9.  Tomato leaf mold
10. Tomato Septoria leaf spot
11. Tomato two-spotted spider mites
12. Tomato target spot
13. Tomato yellow leaf curl virus
14. Tomato mosaic virus
15. Tomato healthy

The exact class identifiers are stored in `models/class_names.json`.
Keep this file consistent with the output order of the trained model.

## Technology stack

-   Python
-   Flask
-   TensorFlow / Keras
-   MobileNetV2
-   NumPy
-   Pillow
-   SQLite
-   Flask-WTF (CSRF protection)
-   Flask-Limiter (rate limiting)
-   python-dotenv (environment configuration)
-   HTML, CSS, and JavaScript

## Project structure

A typical project layout is:

``` text
AI-Crop-Disease-Detection/
│
├── app.py
├── database.py
├── train_model.py
├── requirements.txt
├── .env
│
├── models/
│   ├── crop_disease_model.keras
│   ├── best_crop_model.keras
│   └── class_names.json
│
├── dataset/
│   └── PlantVillage/
│       ├── class_1/
│       ├── class_2/
│       └── ...
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── result.html
│   ├── admin.html
│   └── prediction_detail.html
│
└── static/
    ├── style.css
    └── uploads/
```

The exact folder names may differ in your local copy. The application
expects the trained model and class-name JSON at the paths configured in
`app.py`.

## Requirements

-   Python installed on your computer.
-   Visual Studio Code (recommended for development).
-   The project source files.
-   The trained model file `models/crop_disease_model.keras`.
-   The matching `models/class_names.json`.
-   The PlantVillage dataset if you intend to train the model again.

## Installation and setup (Windows / VS Code)

### 1. Open the project

Open the project folder in Visual Studio Code.

### 2. Create a virtual environment

Open the VS Code terminal (PowerShell) and run:

``` powershell
python -m venv .venv
```

### 3. Activate the virtual environment

``` powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

``` bat
.venv\Scripts\activate.bat
```

### 4. Install dependencies

``` powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The application imports `python-dotenv`, `Flask-WTF`, and
`Flask-Limiter`. If they are not already listed in `requirements.txt`,
install them with:

``` powershell
pip install python-dotenv Flask-WTF Flask-Limiter
```

TensorFlow installation compatibility depends on your Python version and
operating system. If installation fails, use a Python version supported
by the TensorFlow release you plan to install.

### 5. Configure environment variables

Create a `.env` file in the project root. Add your own values:

``` env
SECRET_KEY=replace_with_a_random_secret_key_at_least_32_characters
ADMIN_EMAIL=your_admin_email@example.com
SESSION_COOKIE_SECURE=false
```

Generate a random secret key in the terminal:

``` powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

Copy the generated value into `SECRET_KEY`.

-   Set `ADMIN_EMAIL` to the email address that should have
    administrator access.
-   Keep `.env` private. Do not commit it to a public Git repository or
    share it.
-   For local HTTP development, `SESSION_COOKIE_SECURE=false` is
    appropriate. Set it to `true` only when the application is served
    over HTTPS.

### 6. Check the model files

Make sure these files exist:

``` text
models/crop_disease_model.keras
models/class_names.json
```

`app.py` loads `crop_disease_model.keras` and `class_names.json` when
the application starts. `best_crop_model.keras` is the training
checkpoint/best-validation model artifact; the current application
configuration uses `crop_disease_model.keras`.

### 7. Run the application

``` powershell
python app.py
```

Open this address in your browser:

``` text
http://127.0.0.1:5000/
```

Register an account, log in, and upload a supported crop-leaf image.

## Training the model

If you want to train the model again:

1.  Place the PlantVillage dataset in the dataset directory expected by
    `train_model.py`.
2.  Check the dataset path and training settings near the beginning of
    `train_model.py`.
3.  Activate the virtual environment and ensure the training
    dependencies are installed.
4.  Run:

``` powershell
python train_model.py
```

The training script is configured to create training, validation, and
test splits, train a MobileNetV2-based classifier, and save model and
class-name artifacts. Training can take considerable time and requires
sufficient memory and compute resources.

**Important:** Do not replace the model or class-name file
independently. The model output index and corresponding entry in
`class_names.json` must refer to the same class.

## How prediction works

1.  A logged-in user selects a JPG, JPEG, or PNG image.
2.  The server checks the file type and validates the image.
3.  The image is converted to RGB, resized to 224 × 224 pixels, and
    preprocessed using MobileNetV2 preprocessing.
4.  The trained model predicts class probabilities.
5.  CropAI displays the most likely class, its confidence score, and up
    to three leading predictions.
6.  The prediction is saved to the SQLite database and educational
    information is displayed.

The confidence score represents the model's output for the selected
class. It is not the probability that a field diagnosis is correct.

## Administrator access

The administrator email is configured through `ADMIN_EMAIL` in `.env`.

The admin routes check the email stored in the logged-in session against
the configured administrator email. Only the matching account should be
able to access administrator functions.

Administrator features include viewing prediction statistics and
history, filtering records, viewing prediction details, deleting
prediction records, and exporting prediction data as CSV.

## Database

The application initializes its SQLite database through `database.py`.
User account information and prediction records are stored locally.

Back up the database file before making changes or deleting records. Do
not upload a database containing real user information to a public
repository.

## Troubleshooting

### `SECRET_KEY is missing or too short`

Check that `.env` is in the same project root as `app.py`, and that it
contains a random `SECRET_KEY` with at least 32 characters.

### Model file not found

Check that `models/crop_disease_model.keras` exists and that the
`models` directory is beside `app.py`.

### Class names file not found

Check that `models/class_names.json` exists and contains a valid JSON
list matching the model's output classes.

### `ModuleNotFoundError`

Activate the virtual environment and install the missing package. For
example:

``` powershell
pip install -r requirements.txt
```

For the extra application imports:

``` powershell
pip install python-dotenv Flask-WTF Flask-Limiter
```

### Port 5000 is already in use

Stop the other process using port 5000, or change the port in the
`app.run(...)` section of `app.py`.

### Image upload is rejected

Use a valid JPG, JPEG, or PNG image. The application limits uploads to
10 MB and checks image dimensions and file contents.

## Security notes

-   Use a strong, private `SECRET_KEY`.
-   Never publish `.env`, passwords, database files containing personal
    data, or private user uploads.
-   Keep dependencies updated.
-   Debug mode is enabled in the current development startup code.
    Disable debug mode before deploying the application.
-   Use HTTPS and set `SESSION_COOKIE_SECURE=true` for a production
    deployment.
-   Use a production WSGI server rather than Flask's development server
    for deployment.
-   Apply appropriate backups, access controls, and retention policies
    to uploaded images and database records.

## Limitations

-   The model recognizes only the classes represented in its training
    data.
-   Image quality, lighting, background, crop variety, growth stage, and
    field conditions can affect predictions.
-   A leaf image may not contain enough information to distinguish
    diseases with similar symptoms.
-   A healthy prediction does not guarantee that the entire plant is
    disease-free.
-   The model's test performance on a dataset does not establish
    accuracy on real-world farm images.
-   Treatment guidance is general educational information and may not
    reflect local regulations, crop varieties, or current pest
    conditions.

## License and dataset

Add the license details for your own source code and confirm the terms
of use for any dataset, pretrained weights, or third-party assets before
redistribution.
