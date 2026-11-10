
import os
import json
import uuid
import csv
import io
from functools import wraps
from datetime import timedelta

import numpy as np
import tensorflow as tf

from dotenv import load_dotenv

from flask_wtf.csrf import CSRFProtect

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    Response
)

from werkzeug.utils import secure_filename

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from PIL import Image, UnidentifiedImageError

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from database import (
    initialize_database,
    create_user,
    get_user_by_email,
    save_prediction,
    get_all_predictions,
    get_dashboard_statistics,
    get_prediction_by_id,
    delete_prediction
)


# ==========================================================
# 1. CONFIGURATION AND SECURITY
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, ".env"))

app = Flask(__name__)

SECRET_KEY = os.environ.get("SECRET_KEY", "").strip()

if len(SECRET_KEY) < 32:
    raise RuntimeError(
        "SECRET_KEY is missing or too short. "
        "Add a random 32+ character SECRET_KEY to .env."
    )

app.config["SECRET_KEY"] = SECRET_KEY

ADMIN_EMAIL = os.environ.get(
    "ADMIN_EMAIL", ""
).strip().lower()

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

app.config["SESSION_COOKIE_SECURE"] = (
    os.environ.get(
        "SESSION_COOKIE_SECURE", "false"
    ).strip().lower() in ("1", "true", "yes", "on")
)

app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=30)

app.config["SESSION_REFRESH_EACH_REQUEST"] = True

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

app.config["UPLOAD_FOLDER"] = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

os.makedirs(
    app.config["UPLOAD_FOLDER"],
    exist_ok=True
)

# CSRF protection
csrf = CSRFProtect(app)

# Rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

initialize_database()


# Security headers

@app.after_request
def add_security_headers(response):

    response.headers["X-Content-Type-Options"] = "nosniff"

    response.headers["X-Frame-Options"] = "DENY"

    response.headers["Referrer-Policy"] = (
        "strict-origin-when-cross-origin"
    )

    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=()"
    )

    response.headers["Cache-Control"] = (
        "no-store, no-cache, must-revalidate, private"
    )

    return response


# ==========================================================
# 2. LOAD TRAINED MODEL
# ==========================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "crop_disease_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "class_names.json"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Class names not found: {CLASS_NAMES_PATH}"
    )

print("Loading CropAI model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:

    class_names = json.load(file)

if not isinstance(class_names, list) or not class_names:
    raise ValueError("Invalid class_names.json")

print("CropAI model loaded successfully!")

print("Number of classes:", len(class_names))


# ==========================================================
# 3. ADVANCED DISEASE INFORMATION
# ==========================================================

disease_info = {

    "Pepper__bell___Bacterial_spot": {

        "crop_name": "Bell Pepper",

        "symptoms": [
            "Small dark brown or black spots may appear on leaves.",
            "Lesions can look water-soaked, particularly during humid conditions.",
            "Spots may develop yellow halos around infected areas.",
            "Infections may also cause raised or rough lesions on fruit.",
            "Severe infections can cause leaf yellowing and premature leaf drop."
        ],

        "prevention": [
            "Use certified disease-free seeds and healthy transplants.",
            "Avoid overhead irrigation and working among wet plants.",
            "Remove infected plant debris and manage volunteer pepper plants.",
            "Maintain appropriate spacing to improve airflow.",
            "Rotate with non-host crops where practical.",
            "Use locally registered disease-management products only when recommended."
        ],

        "affected_parts": "Leaves, stems and fruits.",

        "favorable_conditions": (
            "Warm, wet conditions, splashing water and prolonged leaf wetness "
            "can favor bacterial spot development."
        ),

        "monitoring": (
            "Inspect both upper and lower leaf surfaces, especially after "
            "rain or irrigation. Monitor young fruit for raised lesions."
        ),

        "treatment_notes": (
            "Bacterial diseases are difficult to control after establishment. "
            "Use only locally approved products and follow label directions. "
            "Do not assume fungicides will cure bacterial infections."
        )
    },


    "Pepper__bell___healthy": {

        "crop_name": "Bell Pepper",

        "symptoms": [
            "No obvious disease symptoms are indicated by this model class.",
            "Leaves may appear normally green and structurally intact.",
            "No characteristic bacterial spot pattern is indicated."
        ],

        "prevention": [
            "Maintain balanced soil nutrition.",
            "Provide consistent irrigation without waterlogging.",
            "Keep the growing area free from infected plant debris.",
            "Inspect plants regularly for new spots, wilting or discoloration."
        ],

        "affected_parts": "No disease-affected parts identified by this prediction.",

        "favorable_conditions": (
            "Not applicable to a healthy classification."
        ),

        "monitoring": (
            "Continue routine inspection. A healthy model prediction does "
            "not guarantee that the entire plant is disease-free."
        ),

        "treatment_notes": (
            "No disease treatment is indicated by this prediction. "
            "Avoid unnecessary pesticide applications."
        )
    },


    "Potato__Early_blight": {

        "crop_name": "Potato",

        "symptoms": [
            "Early blight commonly begins on older, lower leaves.",
            "Brown or dark lesions may have concentric rings resembling a target.",
            "Yellowing can develop around infected leaf spots.",
            "Lesions may enlarge and merge, causing extensive leaf damage.",
            "Severely affected leaves may dry and fall prematurely."
        ],

        "prevention": [
            "Use healthy seed potatoes and suitable varieties.",
            "Remove infected crop debris after harvest.",
            "Maintain adequate plant spacing for airflow.",
            "Avoid prolonged leaf wetness and unnecessary overhead watering.",
            "Maintain balanced crop nutrition and avoid plant stress.",
            "Rotate crops with non-host plants where practical.",
            "Follow local agricultural guidance for fungicide use when needed."
        ],

        "affected_parts": "Primarily older leaves; stems and tubers may also be affected.",

        "favorable_conditions": (
            "Warm temperatures, leaf wetness, plant stress and infected crop "
            "residues can favor early blight development."
        ),

        "monitoring": (
            "Check older lower leaves for circular brown lesions with "
            "concentric rings. Inspect plants regularly as the crop develops."
        ),

        "treatment_notes": (
            "Confirm the diagnosis before treatment. Use only registered "
            "fungicides where appropriate and follow the label, resistance "
            "management guidance and pre-harvest intervals."
        )
    },


    "Potato__Late_blight": {

        "crop_name": "Potato",

        "symptoms": [
            "Water-soaked or dark green lesions may develop on leaves.",
            "Lesions can quickly become brown or black.",
            "White fungal-like growth may appear on leaf undersides in humid conditions.",
            "Infected leaves and stems can collapse rapidly.",
            "Tubers may develop firm brown or reddish-brown internal discoloration."
        ],

        "prevention": [
            "Use certified disease-free seed potatoes.",
            "Remove volunteer potato plants and infected crop residues.",
            "Avoid overhead watering where possible.",
            "Maintain good field drainage and plant spacing.",
            "Monitor weather and local late-blight advisories.",
            "Remove infected plants and manage them according to local guidance.",
            "Use locally registered protective products when recommended."
        ],

        "affected_parts": "Leaves, stems and tubers.",

        "favorable_conditions": (
            "Cool to moderate temperatures, high humidity, rain and extended "
            "periods of leaf wetness can strongly favor late blight."
        ),

        "monitoring": (
            "Inspect leaves and stems frequently, particularly after wet "
            "weather. Examine tubers for firm brown or reddish discoloration."
        ),

        "treatment_notes": (
            "Late blight can spread rapidly and cause severe crop losses. "
            "Seek prompt local agricultural advice. Follow all fungicide "
            "labels and resistance-management recommendations."
        )
    },


    "Potato__healthy": {

        "crop_name": "Potato",

        "symptoms": [
            "No obvious disease symptoms are indicated by the model.",
            "No characteristic early-blight or late-blight pattern is indicated."
        ],

        "prevention": [
            "Use healthy seed potatoes.",
            "Maintain well-drained soil.",
            "Avoid excessive irrigation.",
            "Remove diseased plant residues.",
            "Inspect leaves and stems regularly."
        ],

        "affected_parts": "No disease-affected parts identified by this prediction.",

        "favorable_conditions": "Not applicable to a healthy classification.",

        "monitoring": (
            "Continue monitoring the crop for unexpected leaf spots, "
            "wilting and stem damage."
        ),

        "treatment_notes": (
            "No disease treatment is indicated by this prediction."
        )
    },


    "Tomato_Bacterial_spot": {

        "crop_name": "Tomato",

        "symptoms": [
            "Small dark brown or black spots can develop on leaves.",
            "Leaf lesions may have yellow halos.",
            "Spots may occur on stems and petioles.",
            "Fruit may develop small raised or scabby lesions.",
            "Severe infections can cause leaf yellowing and defoliation."
        ],

        "prevention": [
            "Use disease-free seeds and healthy transplants.",
            "Avoid handling plants when foliage is wet.",
            "Avoid overhead irrigation where possible.",
            "Remove infected plant debris.",
            "Disinfect tools between plants when contamination is suspected.",
            "Rotate crops with non-host plants.",
            "Maintain airflow through suitable plant spacing."
        ],

        "affected_parts": "Leaves, stems, petioles and fruits.",

        "favorable_conditions": (
            "Warm weather, high humidity, rain splash and prolonged leaf "
            "wetness can encourage bacterial spot."
        ),

        "monitoring": (
            "Inspect leaves and fruit for dark spots, particularly after "
            "rainfall or overhead irrigation."
        ),

        "treatment_notes": (
            "Bacterial spot management relies heavily on prevention and "
            "sanitation. Consult local agricultural experts about registered "
            "products. Do not apply chemicals without checking their labels."
        )
    },


    "Tomato_Early_blight": {

        "crop_name": "Tomato",

        "symptoms": [
            "Small brown spots may first appear on older lower leaves.",
            "Spots often develop concentric rings resembling a target.",
            "Yellow tissue may surround the brown lesions.",
            "Lesions can enlarge and merge into larger damaged areas.",
            "Leaves may turn yellow, dry and fall prematurely.",
            "Stem lesions and fruit lesions near the stem end can also occur."
        ],

        "prevention": [
            "Remove severely infected lower leaves without unnecessarily damaging the plant.",
            "Keep infected crop debris away from healthy plants.",
            "Apply mulch to reduce soil splash onto lower leaves.",
            "Water near the soil rather than wetting the foliage.",
            "Maintain sufficient spacing and airflow.",
            "Rotate crops with non-host plants where practical.",
            "Maintain balanced nutrition and reduce plant stress.",
            "Use locally recommended fungicides only when necessary."
        ],

        "affected_parts": "Older leaves, stems and occasionally fruits.",

        "favorable_conditions": (
            "Warm temperatures, frequent leaf wetness, high humidity, "
            "infected debris and stressed plants can favor early blight."
        ),

        "monitoring": (
            "Check older leaves first. Look for brown target-like rings, "
            "yellow halos and progressive leaf drop. Monitor lower stems "
            "and fruit near the stem attachment."
        ),

        "treatment_notes": (
            "Remove infected material carefully and confirm the diagnosis. "
            "If fungicide treatment is required, use locally registered "
            "products and follow label instructions and pre-harvest intervals."
        )
    },


    "Tomato_Late_blight": {

        "crop_name": "Tomato",

        "symptoms": [
            "Water-soaked pale green or dark lesions may appear on leaves.",
            "Lesions can rapidly enlarge and turn brown or black.",
            "White growth may be visible on leaf undersides in humid weather.",
            "Dark lesions can develop on stems and petioles.",
            "Fruit may develop firm, brown, irregular patches."
        ],

        "prevention": [
            "Use healthy seedlings and disease-free planting material.",
            "Remove infected plant debris and volunteer host plants.",
            "Avoid prolonged leaf wetness.",
            "Provide adequate spacing and ventilation.",
            "Monitor local late-blight alerts and weather conditions.",
            "Inspect plants frequently during cool, wet weather.",
            "Seek local guidance for appropriate disease management."
        ],

        "affected_parts": "Leaves, petioles, stems and fruits.",

        "favorable_conditions": (
            "Cool to moderate temperatures, high humidity, rain and extended "
            "leaf wetness favor late blight."
        ),

        "monitoring": (
            "Inspect the whole plant regularly. Look for rapidly expanding "
            "water-soaked lesions, stem damage and fruit discoloration."
        ),

        "treatment_notes": (
            "Late blight can progress quickly. Seek prompt expert advice "
            "and use only locally registered products according to their labels."
        )
    },


    "Tomato_Leaf_Mold": {

        "crop_name": "Tomato",

        "symptoms": [
            "Pale green or yellow patches can appear on upper leaf surfaces.",
            "Olive-green to brown fuzzy growth may develop underneath leaves.",
            "Affected leaves may curl and dry.",
            "Severe infection can cause premature leaf drop.",
            "Older leaves are often affected first."
        ],

        "prevention": [
            "Improve greenhouse and field ventilation.",
            "Avoid excessive humidity and prolonged leaf wetness.",
            "Provide adequate spacing between plants.",
            "Remove infected leaves and plant debris.",
            "Avoid excessive watering.",
            "Use resistant varieties where available.",
            "Clean greenhouse equipment and growing areas."
        ],

        "affected_parts": "Primarily leaves; severe infection can reduce overall plant vigor.",

        "favorable_conditions": (
            "High humidity, poor ventilation and warm conditions can favor "
            "tomato leaf mold."
        ),

        "monitoring": (
            "Inspect the upper and lower surfaces of leaves. Look for yellow "
            "patches above and olive-colored growth underneath."
        ),

        "treatment_notes": (
            "Improve environmental conditions first. Seek local guidance "
            "before using a registered fungicide."
        )
    },


    "Tomato_Septoria_leaf_spot": {

        "crop_name": "Tomato",

        "symptoms": [
            "Small circular spots appear on leaves.",
            "Lesions often have dark brown margins and pale gray centers.",
            "Tiny dark structures may be visible in lesion centers.",
            "Numerous lesions can cause leaves to turn yellow.",
            "Severely infected lower leaves may dry and fall."
        ],

        "prevention": [
            "Remove infected lower leaves when practical.",
            "Avoid overhead watering.",
            "Use mulch to reduce soil splash.",
            "Remove crop debris after harvest.",
            "Maintain appropriate plant spacing.",
            "Rotate crops with non-host plants.",
            "Disinfect tools where appropriate."
        ],

        "affected_parts": "Primarily leaves, especially lower leaves.",

        "favorable_conditions": (
            "Warm, humid conditions, rain splash and prolonged leaf wetness "
            "can favor Septoria leaf spot."
        ),

        "monitoring": (
            "Check lower leaves for small circular lesions with gray centers "
            "and dark borders."
        ),

        "treatment_notes": (
            "Sanitation and leaf-wetness management are important. "
            "Use only locally registered products when advised."
        )
    },


    "Tomato_Spider_mites_Two_spotted_spider_mite": {

        "crop_name": "Tomato",

        "symptoms": [
            "Tiny yellow or pale feeding spots may appear on leaves.",
            "Leaves can develop a stippled or speckled appearance.",
            "Fine webbing may be visible on leaf undersides and between leaves.",
            "Leaves may become bronze, dry or brittle.",
            "Severe infestations can reduce plant vigor."
        ],

        "prevention": [
            "Inspect leaf undersides regularly using a hand lens.",
            "Monitor for mites and webbing before populations increase.",
            "Reduce excessive dust around growing areas.",
            "Avoid unnecessary broad-spectrum pesticide applications that harm beneficial predators.",
            "Use suitable integrated pest management practices.",
            "Seek local advice about approved mite-control methods."
        ],

        "affected_parts": "Primarily leaves, especially their undersides.",

        "favorable_conditions": (
            "Hot, dry conditions can favor rapid spider mite population growth."
        ),

        "monitoring": (
            "Inspect the undersides of leaves and look for tiny moving mites, "
            "feeding stippling and fine webbing."
        ),

        "treatment_notes": (
            "Spider mites are pests, not fungal diseases. Confirm the presence "
            "of mites before treatment. Follow local pesticide guidance."
        )
    },


    "Tomato__Target_Spot": {

        "crop_name": "Tomato",

        "symptoms": [
            "Brown circular lesions may develop on leaves.",
            "Spots can have concentric rings and a target-like appearance.",
            "Lesions may enlarge and merge.",
            "Dark lesions can also occur on stems and fruit.",
            "Severe infections may cause defoliation."
        ],

        "prevention": [
            "Remove infected crop debris.",
            "Maintain sufficient spacing and airflow.",
            "Avoid prolonged leaf wetness.",
            "Use mulch to reduce soil splash.",
            "Rotate crops where practical.",
            "Monitor the crop regularly for new lesions."
        ],

        "affected_parts": "Leaves, stems and fruits.",

        "favorable_conditions": (
            "Warm temperatures, high humidity and prolonged leaf wetness "
            "can favor target spot."
        ),

        "monitoring": (
            "Inspect leaves for expanding target-like lesions and monitor "
            "stems and fruit for additional damage."
        ),

        "treatment_notes": (
            "Target spot can resemble early blight and other leaf diseases. "
            "Confirm the diagnosis before applying treatment."
        )
    },


    "Tomato__Tomato_YellowLeaf__Curl_Virus": {

        "crop_name": "Tomato",

        "symptoms": [
            "Leaves may turn yellow, especially around the margins.",
            "Leaflets may curl upward and become smaller.",
            "New growth can appear distorted.",
            "Plants may become stunted.",
            "Flower production and fruit setting may be reduced."
        ],

        "prevention": [
            "Use healthy seedlings and resistant varieties where available.",
            "Monitor for whiteflies, which transmit the virus.",
            "Remove severely affected plants according to local guidance.",
            "Manage nearby weed hosts and volunteer tomato plants.",
            "Use integrated whitefly management.",
            "Protect young plants using suitable insect-exclusion methods."
        ],

        "affected_parts": "Leaves, growing tips and the whole plant.",

        "favorable_conditions": (
            "Whitefly activity, susceptible varieties and nearby infected "
            "host plants can contribute to disease spread."
        ),

        "monitoring": (
            "Inspect young leaves for upward curling, yellowing and distorted "
            "growth. Check leaf undersides for whiteflies."
        ),

        "treatment_notes": (
            "There is no curative treatment for an established viral infection. "
            "Focus on prevention and vector management. Seek local expert advice."
        )
    },


    "Tomato__Tomato_mosaic_virus": {

        "crop_name": "Tomato",

        "symptoms": [
            "Leaves may show alternating light and dark green mosaic patterns.",
            "Leaflets can become distorted or narrowed.",
            "Young growth may appear malformed.",
            "Plants may show reduced vigor.",
            "Fruit development and appearance can be affected."
        ],

        "prevention": [
            "Use certified virus-free seeds and healthy transplants.",
            "Wash hands before handling healthy plants after infected plants.",
            "Disinfect tools using an appropriate recommended disinfectant.",
            "Remove infected plants according to local guidance.",
            "Control volunteer tomato plants and weed hosts.",
            "Avoid moving contaminated plant material between growing areas."
        ],

        "affected_parts": "Leaves, growing tips and fruits.",

        "favorable_conditions": (
            "Contaminated tools, hands, plant material and infected host "
            "plants can contribute to mechanical virus transmission."
        ),

        "monitoring": (
            "Look for mosaic patterns, leaf distortion and unusual growth. "
            "Inspect nearby plants for similar symptoms."
        ),

        "treatment_notes": (
            "Viral infections cannot be cured with fungicides. Focus on "
            "sanitation, clean planting material and preventing spread."
        )
    },


    "Tomato_healthy": {

        "crop_name": "Tomato",

        "symptoms": [
            "No obvious disease symptoms are indicated by the model.",
            "No characteristic disease pattern is indicated by this classification.",
            "Leaves may appear normally green and structurally intact."
        ],

        "prevention": [
            "Maintain balanced soil nutrition.",
            "Provide consistent irrigation.",
            "Avoid excessive leaf wetness.",
            "Maintain proper plant spacing.",
            "Remove dead plant debris.",
            "Inspect the crop regularly."
        ],

        "affected_parts": "No disease-affected parts identified by this prediction.",

        "favorable_conditions": (
            "Not applicable to a healthy classification."
        ),

        "monitoring": (
            "Continue regular inspections. A healthy image prediction does "
            "not guarantee that the entire plant is disease-free."
        ),

        "treatment_notes": (
            "No disease treatment is indicated by this prediction. "
            "Avoid unnecessary chemical applications."
        )
    }
}


# ==========================================================
# 4. DISEASE INFORMATION HELPERS
# ==========================================================

def get_disease_information(disease):

    information = disease_info.get(disease)

    if information:
        return information

    return {

        "crop_name": "Unknown crop",

        "symptoms": [
            "Detailed symptoms are not available for this model class."
        ],

        "prevention": [
            "Consult an agricultural expert for identification.",
            "Avoid applying treatments without confirming the disease."
        ],

        "affected_parts": "Not available.",

        "favorable_conditions": "Not available.",

        "monitoring": (
            "Observe the plant and record any visible changes."
        ),

        "treatment_notes": (
            "Confirm the disease before selecting a treatment."
        )
    }


def get_confidence_level(confidence):

    if confidence >= 85:

        return (
            "High model confidence",
            "The model strongly favors this class. "
            "Verify the prediction using visible symptoms."
        )

    elif confidence >= 60:

        return (
            "Moderate model confidence",
            "The model favors this class, but other classes "
            "may also be plausible. Additional inspection is recommended."
        )

    else:

        return (
            "Low model confidence",
            "The prediction is uncertain. Upload a clearer "
            "supported image and seek expert assessment."
        )


# ==========================================================
# 5. LOGIN AND ADMIN SECURITY
# ==========================================================

def login_required(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        if "user_id" not in session:

            flash(
                "Please log in to continue.",
                "error"
            )

            return redirect(url_for("login"))

        return function(*args, **kwargs)

    return wrapper


def admin_required(function):

    @wraps(function)
    @login_required
    def wrapper(*args, **kwargs):

        configured_admin_email = (ADMIN_EMAIL or "").strip().lower()
        logged_in_email = (
            session.get("user_email") or ""
        ).strip().lower()

        if not configured_admin_email:
            return (
                "Admin access is not configured. "
                "Set ADMIN_EMAIL in your .env file and restart CropAI.",
                403
            )

        # The configured admin email is the source of truth.
        # Do not rely on an old role value stored in the browser session.
        if not logged_in_email or logged_in_email != configured_admin_email:
            return (
                "Access denied. Please log in using the administrator email "
                "configured in your .env file.",
                403
            )

        # Refresh the role in the current session.
        session["role"] = "admin"

        return function(*args, **kwargs)

    return wrapper


# ==========================================================
# 6. IMAGE VALIDATION AND PREDICTION
# ==========================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def validate_image(image_path):

    try:

        with Image.open(image_path) as image:

            image.verify()

        with Image.open(image_path) as image:

            if image.width < 100 or image.height < 100:

                raise ValueError(
                    "Image resolution is too small."
                )

            if image.width > 10000 or image.height > 10000:

                raise ValueError(
                    "Image dimensions are too large."
                )

            if image.format not in ("JPEG", "PNG"):

                raise ValueError(
                    "Only JPG, JPEG and PNG images are supported."
                )

    except (UnidentifiedImageError, OSError) as error:

        raise ValueError("The uploaded file is not a valid image.") from error


def predict_disease(image_path):

    validate_image(image_path)

    with Image.open(image_path) as original_image:

        image = original_image.convert("RGB")

        image = image.resize(
            (224, 224),
            Image.Resampling.BILINEAR
        )

        image_array = np.array(
            image,
            dtype=np.float32
        )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_array = preprocess_input(image_array)

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    if len(predictions) != len(class_names):

        raise ValueError(
            "Model output and class names do not match."
        )

    if not np.all(np.isfinite(predictions)):

        raise ValueError(
            "Model returned invalid prediction values."
        )

    # Sort predictions from highest to lowest.
    top_indices = np.argsort(predictions)[::-1][:3]

    top_predictions = []

    for index in top_indices:

        class_name = class_names[int(index)]

        confidence = float(predictions[int(index)]) * 100

        top_predictions.append({

            "disease": class_name,

            "display_name": (
                class_name.replace("_", " ").replace("__", " ").title()
            ),

            "confidence": round(confidence, 2)

        })

    predicted_index = int(top_indices[0])

    disease = class_names[predicted_index]

    confidence = round(
        float(predictions[predicted_index]) * 100,
        2
    )

    return disease, confidence, top_predictions


# ==========================================================
# 7. HOME
# ==========================================================

@app.route("/")
def home():

    if "user_id" in session:

        return redirect(url_for("index"))

    return redirect(url_for("login"))


# ==========================================================
# 8. REGISTRATION
# ==========================================================

@app.route("/register", methods=["GET", "POST"])
@limiter.limit("5 per minute", methods=["POST"])
def register():

    if "user_id" in session:

        return redirect(url_for("index"))

    if request.method == "POST":

        name = request.form.get(
            "name", ""
        ).strip()

        email = request.form.get(
            "email", ""
        ).strip().lower()

        password = request.form.get(
            "password", ""
        )

        confirm_password = request.form.get(
            "confirm_password", ""
        )

        if not name or not email or not password:

            flash(
                "Complete all required fields.",
                "error"
            )

        elif len(name) < 2 or len(name) > 80:

            flash(
                "Name must be between 2 and 80 characters.",
                "error"
            )

        elif len(password) < 12:

            flash(
                "Password must contain at least 12 characters.",
                "error"
            )

        elif password != confirm_password:

            flash(
                "Passwords do not match.",
                "error"
            )

        else:

            password_hash = generate_password_hash(password)

            if create_user(name, email, password_hash):

                flash(
                    "Registration successful. Please log in.",
                    "success"
                )

                return redirect(url_for("login"))

            flash(
                "Email is already registered.",
                "error"
            )

    return render_template("register.html")


# ==========================================================
# 9. LOGIN
# ==========================================================

@app.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per minute", methods=["POST"])
def login():

    if "user_id" in session:

        return redirect(url_for("index"))

    if request.method == "POST":

        email = request.form.get(
            "email", ""
        ).strip().lower()

        password = request.form.get(
            "password", ""
        )

        user = get_user_by_email(email)

        if user and check_password_hash(
            user["password"],
            password
        ):

            session.clear()

            session.permanent = True

            session["user_id"] = user["id"]

            session["user_name"] = user["name"]

            session["user_email"] = (
                user["email"] or ""
            ).strip().lower()

            session["role"] = (
                "admin"
                if ADMIN_EMAIL
                and session["user_email"] == ADMIN_EMAIL.strip().lower()
                else "user"
            )

            flash(
                "Welcome to CropAI!",
                "success"
            )

            return redirect(url_for("index"))

        flash(
            "Invalid email or password.",
            "error"
        )

    return render_template("login.html")


# ==========================================================
# 10. LOGOUT
# ==========================================================

@app.route("/logout")
@login_required
def logout():

    session.clear()

    flash(
        "Logged out successfully.",
        "success"
    )

    return redirect(url_for("login"))


# ==========================================================
# 11. ADVANCED CROP DISEASE PREDICTION
# ==========================================================

@app.route("/dashboard", methods=["GET", "POST"])
@app.route("/predict", methods=["POST"])
@app.route("/home", methods=["GET", "POST"])
@login_required
@limiter.limit("20 per hour", methods=["POST"])
def index():

    if request.method == "POST":

        if "file" not in request.files:

            flash(
                "Please select an image.",
                "error"
            )

            return redirect(url_for("index"))

        file = request.files["file"]

        if file.filename == "":

            flash(
                "No image selected.",
                "error"
            )

            return redirect(url_for("index"))

        if not allowed_file(file.filename):

            flash(
                "Upload JPG, JPEG or PNG images only.",
                "error"
            )

            return redirect(url_for("index"))

        filename = secure_filename(file.filename)

        filename = f"{uuid.uuid4().hex}_{filename}"

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        try:

            file.save(image_path)

            # Generate model prediction and top 3 classes.
            disease, confidence, top_predictions = predict_disease(
                image_path
            )

            # Retrieve detailed information.
            information = get_disease_information(disease)

            confidence_label, confidence_message = (
                get_confidence_level(confidence)
            )

            # Save prediction in database.
            save_prediction(
                filename,
                disease,
                confidence
            )

            # Display additional warning for uncertain predictions.
            image_warning = None

            if confidence < 60:

                image_warning = (
                    "The model has low confidence in this prediction. "
                    "The image may be unclear, unsupported or visually "
                    "different from the training data. Do not rely on "
                    "this prediction for treatment decisions."
                )

            elif confidence < 85:

                image_warning = (
                    "The model has moderate confidence. Compare the "
                    "visible symptoms with the disease information "
                    "and consider expert verification."
                )

            # Display result page.
            return render_template(

                "result.html",

                image_path=filename,

                disease=disease,

                confidence=confidence,

                crop_name=information["crop_name"],

                symptoms=information["symptoms"],

                prevention=information["prevention"],

                affected_parts=information["affected_parts"],

                favorable_conditions=information["favorable_conditions"],

                monitoring=information["monitoring"],

                treatment_notes=information["treatment_notes"],

                top_predictions=top_predictions,

                confidence_label=confidence_label,

                confidence_message=confidence_message,

                image_warning=image_warning

            )

        except (UnidentifiedImageError, OSError, ValueError) as error:

            if os.path.exists(image_path):

                os.remove(image_path)

            app.logger.warning(
                "Invalid uploaded image: %s",
                error
            )

            flash(
                str(error),
                "error"
            )

            return redirect(url_for("index"))

        except Exception:

            app.logger.exception(
                "Prediction failed"
            )

            if os.path.exists(image_path):

                os.remove(image_path)

            flash(
                "Unable to process the image. Please try again.",
                "error"
            )

            return redirect(url_for("index"))

    return render_template(
        "index.html",
        user_name=session.get("user_name")
    )


# ==========================================================
# 12. ADMIN DASHBOARD
# ==========================================================

@app.route("/admin")
@admin_required
def admin_dashboard():

    all_records = get_all_predictions() or []

    stats = get_dashboard_statistics()

    search = request.args.get(
        "search", ""
    ).strip()

    status = request.args.get(
        "status", "all"
    ).lower()

    if status not in ("all", "healthy", "diseased"):

        status = "all"

    selected_date = request.args.get(
        "date", ""
    ).strip()

    try:

        page = max(
            1,
            int(request.args.get("page", 1))
        )

    except (ValueError, TypeError):

        page = 1

    per_page = 10

    records = [

        record for record in all_records

        if search.lower() in
        record["disease"].replace("_", " ").lower()

    ]

    if status == "healthy":

        records = [

            record for record in records

            if "healthy" in record["disease"].lower()

        ]

    elif status == "diseased":

        records = [

            record for record in records

            if "healthy" not in record["disease"].lower()

        ]

    if selected_date:

        records = [

            record for record in records

            if record["created_at"][:10] == selected_date

        ]

    total_filtered = len(records)

    total_pages = max(
        1,
        (total_filtered + per_page - 1) // per_page
    )

    page = min(page, total_pages)

    start = (page - 1) * per_page

    paginated_records = records[
        start:start + per_page
    ]

    return render_template(

        "admin.html",

        stats=stats,

        records=paginated_records,

        user_name=session.get("user_name", "Admin"),

        search=search,

        status=status,

        selected_date=selected_date,

        page=page,

        total_pages=total_pages,

        total_filtered=total_filtered

    )


# ==========================================================
# 13. VIEW PREDICTION DETAILS
# ==========================================================

@app.route("/admin/prediction/<int:prediction_id>")
@admin_required
def prediction_details(prediction_id):

    record = get_prediction_by_id(
        prediction_id
    )

    if not record:

        return "Prediction record not found.", 404

    filename = record["filename"]

    if (
        not filename
        or os.path.basename(filename) != filename
    ):

        return "Invalid image filename.", 400

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    if not os.path.isfile(image_path):

        return "Prediction image not found.", 404

    disease = record["disease"]

    information = get_disease_information(disease)

    return render_template(

        "prediction_detail.html",

        record=record,

        image_filename=filename,

        disease=disease.replace("_", " "),

        symptoms=information["symptoms"],

        prevention=information["prevention"],

        user_name=session.get("user_name", "Admin")

    )


# ==========================================================
# 14. DELETE PREDICTION
# ==========================================================

@app.route(
    "/admin/delete-prediction/<int:prediction_id>",
    methods=["POST"]
)
@admin_required
def delete_prediction_route(prediction_id):

    record = get_prediction_by_id(
        prediction_id
    )

    if not record:

        flash(
            "Prediction record not found.",
            "error"
        )

    else:

        filename = record.get(
            "filename", ""
        )

        delete_prediction(prediction_id)

        if (
            filename
            and os.path.basename(filename) == filename
        ):

            image_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            try:

                if os.path.isfile(image_path):

                    os.remove(image_path)

            except OSError:

                app.logger.exception(
                    "Image deletion failed"
                )

        flash(
            f"Prediction #{prediction_id} deleted successfully.",
            "success"
        )

    search = request.form.get(
        "search", ""
    )

    status = request.form.get(
        "status", "all"
    )

    selected_date = request.form.get(
        "date", ""
    )

    try:

        page = max(
            1,
            int(request.form.get("page", 1))
        )

    except (ValueError, TypeError):

        page = 1

    return redirect(

        url_for(

            "admin_dashboard",

            search=search,

            status=status,

            date=selected_date,

            page=page

        ) + "#history"

    )


# ==========================================================
# 15. CSV EXPORT
# ==========================================================

@app.route("/export")
@admin_required
def export_predictions():

    records = get_all_predictions() or []

    output = io.StringIO()

    writer = csv.writer(output)

    writer.writerow([

        "ID",

        "Filename",

        "Disease",

        "Confidence",

        "Created At"

    ])

    for record in records:

        writer.writerow([

            record["id"],

            record["filename"],

            record["disease"],

            record["confidence"],

            record["created_at"]

        ])

    response = Response(
        output.getvalue(),
        mimetype="text/csv"
    )

    response.headers["Content-Disposition"] = (
        "attachment; filename=cropai_predictions.csv"
    )

    return response


# ==========================================================
# 16. ERROR HANDLING
# ==========================================================

@app.errorhandler(413)
def file_too_large(error):

    flash(
        "Image size must be below 10 MB.",
        "error"
    )

    return redirect(url_for("index"))


@app.errorhandler(404)
def page_not_found(error):

    return (
        "Page not found. Please check the URL.",
        404
    )


# ==========================================================
# 17. START APPLICATION
# ==========================================================

if __name__ == "__main__":

    print("===================================")

    print("          CropAI System")

    print("===================================")

    print(
        "Admin configured:",
        bool(ADMIN_EMAIL)
    )

    print(
        "Number of disease classes:",
        len(class_names)
    )

    print(
        "Open: http://127.0.0.1:5000"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
