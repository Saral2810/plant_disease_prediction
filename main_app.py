import streamlit as st
import numpy as np
import cv2
import time
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("plant_disease_model.h5")

# Class labels for plant diseases
CLASS_NAMES = ['Tomato-Bacterial_spot', 'Potato-Barly_blight', 'Corn-Common_rust', 'Apple-Apple_scab', 
            'Apple-Black_rot', 'Apple-Cedar_apple_rust', 'Apple-healthy', 'Blueberry-healthy', 
            'Cherry-Powdery_mildew', 'Cherry-healthy', 'Corn-Gray_leaf_spot', 'Corn-Common_rust', 
            'Corn-Leaf_Blight', 'Corn-healthy', 'Grape-Black_rot', 'Grape-Esca', 'Grape-Leaf_blight', 
            'Grape-healthy', 'Orange-Citrus_greening', 'Peach-Bacterial_spot', 'Peach-healthy', 
            'Pepper-Bacterial_spot', 'Pepper-healthy', 'Potato-Early_blight', 'Potato-Late_blight', 
            'Potato-healthy', 'Raspberry-healthy', 'Soybean-healthy', 'Squash-Powdery_mildew', 
            'Strawberry-Leaf_scorch', 'Strawberry-healthy', 'Tomato-Bacterial_spot', 'Tomato-Early_blight', 
            'Tomato-Late_blight', 'Tomato-Leaf_Mold', 'Tomato-Septoria_leaf_spot', 'Tomato-Spider_mites', 
            'Tomato-Target_Spot', 'Tomato-Yellow_Leaf_Curl_Virus', 'Tomato-mosaic_virus', 'Tomato-healthy']

# Language translations with all major Indian languages
TRANSLATIONS = {
    "en": {
        "title": "🌱 PLANT DISEASE RECOGNITION SYSTEM",
        "home_title": "AI-Powered Plant Disease Detection",
        "welcome": "Welcome!",
        "description": "This system helps identify plant diseases using **AI-powered image analysis**.",
        "how_it_works": "### How It Works:",
        "step1": "1. **Upload an Image**: Go to the Disease Recognition page.",
        "step2": "2. **AI Processing**: The model detects potential diseases.",
        "step3": "3. **Results**: Get accurate disease predictions instantly!",
        "start": "👉 *Start by clicking on the Disease Recognition tab!*",
        "about_title": "📜 About This Project",
        "developed_by": "Developed by:",
        "team": "\n- **Saral Singhal**\n- **Aditi Shindadkar**\n- **Om Raj**\n- **Tarpita Karnam**\n- **Pranjal Karan**",
        "recognition_title": "🔬 Disease Recognition",
        "upload_text": "📸 Upload a Plant Leaf Image:",
        "predict_button": "🔍 Predict Disease",
        "analyzing": "🔄 Analyzing Image...",
        "result_disease": "✅ Disease Detected:",
        "result_plant": "🍃 Plant Type:",
        "select_page": "Select Page",
        "home": "Home",
        "about": "About",
        "disease_recognition": "Disease Recognition"
    },
    "hi": {
        "title": "🌱 पौध रोग पहचान प्रणाली",
        "home_title": "एआई-संचालित पौध रोग पहचान",
        "welcome": "स्वागत है!",
        "description": "यह प्रणाली **एआई-संचालित छवि विश्लेषण** का उपयोग करके पौधों के रोगों की पहचान करने में मदद करती है।",
        "how_it_works": "### यह कैसे काम करता है:",
        "step1": "1. **एक छवि अपलोड करें**: रोग पहचान पृष्ठ पर जाएं।",
        "step2": "2. **एआई प्रसंस्करण**: मॉडल संभावित रोगों का पता लगाता है।",
        "step3": "3. **परिणाम**: तुरंत सटीक रोग भविष्यवाणियां प्राप्त करें!",
        "start": "👉 *रोग पहचान टैब पर क्लिक करके शुरू करें!*",
        "about_title": "📜 इस परियोजना के बारे में",
        "developed_by": "द्वारा विकसित:",
        "team": "- **सरल सिंघल**\n- **अदिति शिंदाडकर**\n- **ओम राज**\n- **तर्पिता कर्णम**\n- **प्रांजल करण**",
        "recognition_title": "🔬 रोग पहचान",
        "upload_text": "📸 एक पौधे की पत्ती की छवि अपलोड करें:",
        "predict_button": "🔍 रोग की भविष्यवाणी करें",
        "analyzing": "🔄 छवि का विश्लेषण...",
        "result_disease": "✅ रोग का पता चला:",
        "result_plant": "🍃 पौधे का प्रकार:",
        "select_page": "पृष्ठ चुनें",
        "home": "होम",
        "about": "जानकारी",
        "disease_recognition": "रोग पहचान"
    },
    "bn": {  # Bengali
        "title": "🌱 উদ্ভিদ রোগ চিন্হিতকরণ ব্যবস্থা",
        "home_title": "এআই-চালিত উদ্ভিদ রোগ সনাক্তকরণ",
        "welcome": "স্বাগতম!",
        "description": "এই সিস্টেমটি **এআই-চালিত ইমেজ বিশ্লেষণ** ব্যবহার করে উদ্ভিদের রোগ সনাক্ত করতে সহায়তা করে।",
        "how_it_works": "### এটি কিভাবে কাজ করে:",
        "step1": "1. **একটি ছবি আপলোড করুন**: রোগ চিন্হিতকরণ পৃষ্ঠায় যান।",
        "step2": "2. **এআই প্রক্রিয়াকরণ**: মডেল সম্ভাব্য রোগ সনাক্ত করে।",
        "step3": "3. **ফলাফল**: অবিলম্বে সঠিক রোগের পূর্বাভাস পান!",
        "start": "👉 *রোগ চিন্হিতকরণ ট্যাবে ক্লিক করে শুরু করুন!*",
        "about_title": "📜 এই প্রকল্প সম্পর্কে",
        "developed_by": "দ্বারা উন্নত:",
        "team": "- **সরল সিংহল**\n- **অদিতি শিন্দাড়কর**\n- **ওম রাজ**\n- **তর্পিতা কর্ণম**\n- **প্রাঞ্জল করণ**",
        "recognition_title": "🔬 রোগ চিন্হিতকরণ",
        "upload_text": "📸 একটি উদ্ভিদ পাতার ছবি আপলোড করুন:",
        "predict_button": "🔍 রোগের পূর্বাভাস দিন",
        "analyzing": "🔄 ছবি বিশ্লেষণ করা হচ্ছে...",
        "result_disease": "✅ রোগ সনাক্ত হয়েছে:",
        "result_plant": "🍃 উদ্ভিদের ধরন:",
        "select_page": "পৃষ্ঠা নির্বাচন করুন",
        "home": "হোম",
        "about": "সম্পর্কে",
        "disease_recognition": "রোগ চিন্হিতকরণ"
    },
    "ta": {  # Tamil
        "title": "🌱 தாவர நோய் அங்கீகார அமைப்பு",
        "home_title": "AI-இயக்கப்படும் தாவர நோய் கண்டறிதல்",
        "welcome": "வரவேற்கிறோம்!",
        "description": "இந்த அமைப்பு **AI-இயக்கப்படும் பட பகுப்பாய்வு** பயன்படுத்தி தாவர நோய்களை அடையாளம் காண உதவுகிறது.",
        "how_it_works": "### இது எவ்வாறு செயல்படுகிறது:",
        "step1": "1. **படத்தை பதிவேற்றவும்**: நோய் அங்கீகார பக்கத்திற்குச் செல்லவும்.",
        "step2": "2. **AI செயலாக்கம்**: மாதிரி சாத்தியமான நோய்களை கண்டறியும்.",
        "step3": "3. **முடிவுகள்**: துல்லியமான நோய் கணிப்புகளை உடனடியாகப் பெறவும்!",
        "start": "👉 *நோய் அங்கீகார தாவலைக் கிளிக் செய்வதன் மூலம் தொடங்கவும்!*",
        "about_title": "📜 இந்த திட்டம் பற்றி",
        "developed_by": "மூலம் உருவாக்கப்பட்டது:",
        "team": "- **சரல் சிங்ல்**\n- **அதிதி ஷிண்டாட்கர்**\n- **ஓம் ராஜ்**\n- **தார்பிதா கர்ணம்**\n- **பிராஞ்சல் கரண்**",
        "recognition_title": "🔬 நோய் அங்கீகாரம்",
        "upload_text": "📸 ஒரு தாவர இலை படத்தை பதிவேற்றவும்:",
        "predict_button": "🔍 நோயை கணிக்கவும்",
        "analyzing": "🔄 படத்தை பகுப்பாய்வு செய்கிறது...",
        "result_disease": "✅ நோய் கண்டறியப்பட்டது:",
        "result_plant": "🍃 தாவர வகை:",
        "select_page": "பக்கத்தை தேர்ந்தெடுக்கவும்",
        "home": "முகப்பு",
        "about": "பற்றி",
        "disease_recognition": "நோய் அங்கீகாரம்"
    },
    "te": {  # Telugu
        "title": "🌱 మొక్కల వ్యాధి గుర్తింపు వ్యవస్థ",
        "home_title": "AI-శక్తితో మొక్కల వ్యాధి గుర్తింపు",
        "welcome": "స్వాగతం!",
        "description": "ఈ వ్యవస్థ **AI-శక్తితో ఇమేజ్ విశ్లేషణ** ఉపయోగించి మొక్కల వ్యాధులను గుర్తించడంలో సహాయపడుతుంది.",
        "how_it_works": "### ఇది ఎలా పని చేస్తుంది:",
        "step1": "1. **ఒక ఇమేజ్ అప్లోడ్ చేయండి**: వ్యాధి గుర్తింపు పేజీకి వెళ్లండి.",
        "step2": "2. **AI ప్రాసెసింగ్**: మోడల్ సంభావ్య వ్యాధులను గుర్తిస్తుంది.",
        "step3": "3. **ఫలితాలు**: ఖచ్చితమైన వ్యాధి అంచనాలను తక్షణమే పొందండి!",
        "start": "👉 *వ్యాధి గుర్తింపు ట్యాబ్ పై క్లిక్ చేయడం ద్వారా ప్రారంభించండి!*",
        "about_title": "📜 ఈ ప్రాజెక్ట్ గురించి",
        "developed_by": "ద్వారా అభివృద్ధి చేయబడింది:",
        "team": "- **సరల్ సింఘల్**\n- **అదితి షిండాడ్కర్**\n- **ఓం రాజ్**\n- **తర్పిత కర్నం**\n- **ప్రాంజల్ కరణ్**",
        "recognition_title": "🔬 వ్యాధి గుర్తింపు",
        "upload_text": "📸 ఒక మొక్క ఆకు ఇమేజ్ అప్లోడ్ చేయండి:",
        "predict_button": "🔍 వ్యాధిని అంచనా వేయండి",
        "analyzing": "🔄 ఇమేజ్ విశ్లేషిస్తోంది...",
        "result_disease": "✅ వ్యాధి గుర్తించబడింది:",
        "result_plant": "🍃 మొక్క రకం:",
        "select_page": "పేజీని ఎంచుకోండి",
        "home": "హోమ్",
        "about": "గురించి",
        "disease_recognition": "వ్యాధి గుర్తింపు"
    },
    "mr": {  # Marathi
        "title": "🌱 वनस्पती रोग ओळख प्रणाली",
        "home_title": "AI-चालित वनस्पती रोग ओळख",
        "welcome": "स्वागत आहे!",
        "description": "ही प्रणाली **AI-चालित प्रतिमा विश्लेषण** वापरून वनस्पतींच्या रोगांची ओळख करण्यास मदत करते.",
        "how_it_works": "### हे कसे काम करते:",
        "step1": "1. **एक प्रतिमा अपलोड करा**: रोग ओळख पृष्ठावर जा.",
        "step2": "2. **AI प्रक्रिया**: मॉडेल संभाव्य रोग ओळखते.",
        "step3": "3. **निकाल**: अचूक रोग अंदाज त्वरित मिळवा!",
        "start": "👉 *रोग ओळख टॅब वर क्लिक करून प्रारंभ करा!*",
        "about_title": "📜 या प्रकल्पाबद्दल",
        "developed_by": "द्वारा विकसित:",
        "team": "- **सरल सिंगल**\n- **अदिती शिंदाडकर**\n- **ओम राज**\n- **तर्पिता कर्णम**\n- **प्रांजल करण**",
        "recognition_title": "🔬 रोग ओळख",
        "upload_text": "📸 एक वनस्पती पानाची प्रतिमा अपलोड करा:",
        "predict_button": "🔍 रोगाचा अंदाज लावा",
        "analyzing": "🔄 प्रतिमेचे विश्लेषण करत आहे...",
        "result_disease": "✅ रोग ओळखला गेला:",
        "result_plant": "🍃 वनस्पती प्रकार:",
        "select_page": "पृष्ठ निवडा",
        "home": "मुख्यपृष्ठ",
        "about": "विषयी",
        "disease_recognition": "रोग ओळख"
    },
    "gu": {  # Gujarati
        "title": "🌱 છોડ રોગ ઓળખ સિસ્ટમ",
        "home_title": "AI-સક્ષમ છોડ રોગ શોધ",
        "welcome": "સ્વાગત છે!",
        "description": "આ સિસ્ટમ **AI-સક્ષમ ઇમેજ વિશ્લેષણ** નો ઉપયોગ કરીને છોડના રોગોને ઓળખવામાં મદદ કરે છે.",
        "how_it_works": "### તે કેવી રીતે કામ કરે છે:",
        "step1": "1. **એક ઇમેજ અપલોડ કરો**: રોગ ઓળખ પૃષ્ઠ પર જાઓ.",
        "step2": "2. **AI પ્રક્રિયા**: મોડેલ સંભવિત રોગોને ઓળખે છે.",
        "step3": "3. **પરિણામો**: ચોક્કસ રોગ આગાહીઓ તરત જ મેળવો!",
        "start": "👉 *રોગ ઓળખ ટૅબ પર ક્લિક કરીને શરૂ કરો!*",
        "about_title": "📜 આ પ્રોજેક્ટ વિશે",
        "developed_by": "દ્વારા વિકસિત:",
        "team": "- **સરલ સિંઘલ**\n- **અદિતિ શિંડાડકર**\n- **ઓમ રાજ**\n- **તર્પિતા કર્ણમ**\n- **પ્રાંજલ કરણ**",
        "recognition_title": "🔬 રોગ ઓળખ",
        "upload_text": "📸 એક છોડ પાંદડાની ઇમેજ અપલોડ કરો:",
        "predict_button": "🔍 રોગની આગાહી કરો",
        "analyzing": "🔄 ઇમેજનું વિશ્લેષણ કરી રહ્યું છે...",
        "result_disease": "✅ રોગ શોધી કાઢ્યો:",
        "result_plant": "🍃 છોડ પ્રકાર:",
        "select_page": "પૃષ્ઠ પસંદ કરો",
        "home": "હોમ",
        "about": "વિશે",
        "disease_recognition": "રોગ ઓળખ"
    },
    "kn": {  # Kannada
        "title": "🌱 ಸಸ್ಯ ರೋಗ ಗುರುತಿಸುವ ವ್ಯವಸ್ಥೆ",
        "home_title": "AI-ಚಾಲಿತ ಸಸ್ಯ ರೋಗ ಪತ್ತೆ",
        "welcome": "ಸ್ವಾಗತ!",
        "description": "ಈ ವ್ಯವಸ್ಥೆಯು **AI-ಚಾಲಿತ ಚಿತ್ರ ವಿಶ್ಲೇಷಣೆ** ಬಳಸಿ ಸಸ್ಯ ರೋಗಗಳನ್ನು ಗುರುತಿಸಲು ಸಹಾಯ ಮಾಡುತ್ತದೆ.",
        "how_it_works": "### ಇದು ಹೇಗೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ:",
        "step1": "1. **ಚಿತ್ರವನ್ನು ಅಪ್ಲೋಡ್ ಮಾಡಿ**: ರೋಗ ಗುರುತಿಸುವ ಪುಟಕ್ಕೆ ಹೋಗಿ.",
        "step2": "2. **AI ಸಂಸ್ಕರಣೆ**: ಮಾದರಿ ಸಂಭಾವ್ಯ ರೋಗಗಳನ್ನು ಗುರುತಿಸುತ್ತದೆ.",
        "step3": "3. **ಫಲಿತಾಂಶಗಳು**: ನಿಖರವಾದ ರೋಗ ಊಹೆಗಳನ್ನು ತಕ್ಷಣ ಪಡೆಯಿರಿ!",
        "start": "👉 *ರೋಗ ಗುರುತಿಸುವ ಟ್ಯಾಬ್ ಕ್ಲಿಕ್ ಮಾಡುವ ಮೂಲಕ ಪ್ರಾರಂಭಿಸಿ!*",
        "about_title": "📜 ಈ ಯೋಜನೆಯ ಬಗ್ಗೆ",
        "developed_by": "ಅಭಿವೃದ್ಧಿಪಡಿಸಿದವರು:",
        "team": "- **ಸರಳ್ ಸಿಂಗಲ್**\n- **ಅದಿತಿ ಶಿಂಡಾಡ್ಕರ್**\n- **ಓಂ ರಾಜ್**\n- **ತರ್ಪಿತಾ ಕರ್ಣಮ್**\n- **ಪ್ರಾಂಜಲ್ ಕರಣ್**",
        "recognition_title": "🔬 ರೋಗ ಗುರುತಿಸುವಿಕೆ",
        "upload_text": "📸 ಸಸ್ಯದ ಎಲೆಯ ಚಿತ್ರವನ್ನು ಅಪ್ಲೋಡ್ ಮಾಡಿ:",
        "predict_button": "🔍 ರೋಗವನ್ನು ಊಹಿಸಿ",
        "analyzing": "🔄 ಚಿತ್ರವನ್ನು ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...",
        "result_disease": "✅ ರೋಗ ಪತ್ತೆಯಾಗಿದೆ:",
        "result_plant": "🍃 ಸಸ್ಯ ಪ್ರಕಾರ:",
        "select_page": "ಪುಟವನ್ನು ಆರಿಸಿ",
        "home": "ಮುಖಪುಟ",
        "about": "ಬಗ್ಗೆ",
        "disease_recognition": "ರೋಗ ಗುರುತಿಸುವಿಕೆ"
    },
    "ml": {  # Malayalam
        "title": "🌱 സസ്യ രോഗ തിരിച്ചറിയൽ സിസ്റ്റം",
        "home_title": "AI-പ്രവർത്തിത സസ്യ രോഗ ഡിറ്റക്ഷൻ",
        "welcome": "സ്വാഗതം!",
        "description": "ഈ സിസ്റ്റം **AI-പ്രവർത്തിത ഇമേജ് വിശകലനം** ഉപയോഗിച്ച് സസ്യ രോഗങ്ങൾ തിരിച്ചറിയാൻ സഹായിക്കുന്നു.",
        "how_it_works": "### ഇത് എങ്ങനെ പ്രവർത്തിക്കുന്നു:",
        "step1": "1. **ഒരു ഇമേജ് അപ്ലോഡ് ചെയ്യുക**: രോഗ തിരിച്ചറിയൽ പേജിലേക്ക് പോകുക.",
        "step2": "2. **AI പ്രോസസ്സിംഗ്**: മോഡൽ സാധ്യതയുള്ള രോഗങ്ങൾ തിരിച്ചറിയുന്നു.",
        "step3": "3. **ഫലങ്ങൾ**: കൃത്യമായ രോഗ പ്രവചനങ്ങൾ ഉടനടി നേടുക!",
        "start": "👉 *രോഗ തിരിച്ചറിയൽ ടാബ് ക്ലിക്ക് ചെയ്ത് ആരംഭിക്കുക!*",
        "about_title": "📜 ഈ പ്രോജക്റ്റിനെ കുറിച്ച്",
        "developed_by": "വികസിപ്പിച്ചത്:",
        "team": "- **സരൽ സിംഗൽ**\n- **അദിതി ഷിണ്ടാഡ്കർ**\n- **ഓം രാജ്**\n- **തർപ്പിത കർണം**\n- **പ്രാഞ്ചൽ കരൺ**",
        "recognition_title": "🔬 രോഗ തിരിച്ചറിയൽ",
        "upload_text": "📸 ഒരു സസ്യ ഇല ഇമേജ് അപ്ലോഡ് ചെയ്യുക:",
        "predict_button": "🔍 രോഗം പ്രവചിക്കുക",
        "analyzing": "🔄 ഇമേജ് വിശകലനം ചെയ്യുന്നു...",
        "result_disease": "✅ രോഗം കണ്ടെത്തി:",
        "result_plant": "🍃 സസ്യ തരം:",
        "select_page": "പേജ് തിരഞ്ഞെടുക്കുക",
        "home": "ഹോം",
        "about": "വിവരണം",
        "disease_recognition": "രോഗ തിരിച്ചറിയൽ"
    },
    "pa": {  # Punjabi
        "title": "🌱 ਪੌਦਾ ਰੋਗ ਪਛਾਣ ਸਿਸਟਮ",
        "home_title": "AI-ਸੰਚਾਲਿਤ ਪੌਦਾ ਰੋਗ ਖੋਜ",
        "welcome": "ਜੀ ਆਇਆਂ ਨੂੰ!",
        "description": "ਇਹ ਸਿਸਟਮ **AI-ਸੰਚਾਲਿਤ ਚਿੱਤਰ ਵਿਸ਼ਲੇਸ਼ਣ** ਦੀ ਵਰਤੋਂ ਕਰਕੇ ਪੌਦਿਆਂ ਦੀਆਂ ਬਿਮਾਰੀਆਂ ਦੀ ਪਛਾਣ ਕਰਨ ਵਿੱਚ ਮਦਦ ਕਰਦਾ ਹੈ.",
        "how_it_works": "### ਇਹ ਕਿਵੇਂ ਕੰਮ ਕਰਦਾ ਹੈ:",
        "step1": "1. **ਇੱਕ ਚਿੱਤਰ ਅੱਪਲੋਡ ਕਰੋ**: ਰੋਗ ਪਛਾਣ ਪੰਨੇ 'ਤੇ ਜਾਓ.",
        "step2": "2. **AI ਪ੍ਰਕਿਰਿਆ**: ਮਾਡਲ ਸੰਭਾਵੀ ਰੋਗਾਂ ਦੀ ਪਛਾਣ ਕਰਦਾ ਹੈ.",
        "step3": "3. **ਨਤੀਜੇ**: ਸਹੀ ਰੋਗ ਭਵਿੱਖਬਾਣੀ ਤੁਰੰਤ ਪ੍ਰਾਪਤ ਕਰੋ!",
        "start": "👉 *ਰੋਗ ਪਛਾਣ ਟੈਬ 'ਤੇ ਕਲਿੱਕ ਕਰਕੇ ਸ਼ੁਰੂ ਕਰੋ!*",
        "about_title": "📜 ਇਸ ਪ੍ਰੋਜੈਕਟ ਬਾਰੇ",
        "developed_by": "ਦੁਆਰਾ ਵਿਕਸਿਤ:",
        "team": "- **ਸਰਲ ਸਿੰਘਲ**\n- **ਅਦਿਤੀ ਸ਼ਿੰਡਾਡਕਰ**\n- **ਓਮ ਰਾਜ**\n- **ਤਰਪਿਤਾ ਕਰਨਮ**\n- **ਪ੍ਰਾਂਜਲ ਕਰਨ**",
        "recognition_title": "🔬 ਰੋਗ ਪਛਾਣ",
        "upload_text": "📸 ਇੱਕ ਪੌਦੇ ਦੀ ਪੱਤੀ ਦੀ ਚਿੱਤਰ ਅੱਪਲੋਡ ਕਰੋ:",
        "predict_button": "🔍 ਰੋਗ ਦੀ ਭਵਿੱਖਬਾਣੀ ਕਰੋ",
        "analyzing": "🔄 ਚਿੱਤਰ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕੀਤਾ ਜਾ ਰਿਹਾ ਹੈ...",
        "result_disease": "✅ ਰੋਗ ਦੀ ਪਛਾਣ:",
        "result_plant": "🍃 ਪੌਦੇ ਦੀ ਕਿਸਮ:",
        "select_page": "ਪੰਨਾ ਚੁਣੋ",
        "home": "ਹੋਮ",
        "about": "ਬਾਰੇ",
        "disease_recognition": "ਰੋਗ ਪਛਾਣ"
    },
    "or": {  # Odia
        "title": "🌱 ଉଦ୍ଭିଦ ରୋଗ ଚିହ୍ନଟ ପ୍ରଣାଳୀ",
        "home_title": "AI-ଚାଳିତ ଉଦ୍ଭିଦ ରୋଗ ଚିହ୍ନଟ",
        "welcome": "ସ୍ୱାଗତ!",
        "description": "ଏହି ପ୍ରଣାଳୀ **AI-ଚାଳିତ ପ୍ରତିଛବି ବିଶ୍ଳେଷଣ** ବ୍ୟବହାର କରି ଉଦ୍ଭିଦ ରୋଗଗୁଡ଼ିକୁ ଚିହ୍ନଟ କରିବାରେ ସାହାଯ୍ୟ କରେ।",
        "how_it_works": "### ଏହା କିପରି କାମ କରେ:",
        "step1": "1. **ଏକ ପ୍ରତିଛବି ଅପଲୋଡ୍ କରନ୍ତୁ**: ରୋଗ ଚିହ୍ନଟ ପୃଷ୍ଠାକୁ ଯାଆନ୍ତୁ।",
        "step2": "2. **AI ପ୍ରକ୍ରିୟାକରଣ**: ମଡେଲ ସମ୍ଭାବ୍ୟ ରୋଗଗୁଡ଼ିକୁ ଚିହ୍ନଟ କରେ।",
        "step3": "3. **ଫଳାଫଳ**: ସଠିକ୍ ରୋଗ ପୂର୍ବାନୁମାନ ତୁରନ୍ତ ପ୍ରାପ୍ତ କରନ୍ତୁ!",
        "start": "👉 *ରୋଗ ଚିହ୍ନଟ ଟ୍ୟାବ୍ ଉପରେ କ୍ଲିକ୍ କରି ଆରମ୍ଭ କରନ୍ତୁ!*",
        "about_title": "📜 ଏହି ପ୍ରକଳ୍ପ ବିଷୟରେ",
        "developed_by": "ଦ୍ୱାରା ବିକଶିତ:",
        "team": "- **ସରଳ ସିଂହଲ**\n- **ଅଦିତି ଶିଣ୍ଡାଡକର**\n- **ଓମ ରାଜ**\n- **ତର୍ପିତା କର୍ଣ୍ଣମ**\n- **ପ୍ରାଞ୍ଜଲ କରଣ**",
        "recognition_title": "🔬 ରୋଗ ଚିହ୍ନଟ",
        "upload_text": "📸 ଏକ ଉଦ୍ଭିଦ ପତ୍ର ପ୍ରତିଛବି ଅପଲୋଡ୍ କରନ୍ତୁ:",
        "predict_button": "🔍 ରୋଗ ପୂର୍ବାନୁମାନ କରନ୍ତୁ",
        "analyzing": "🔄 ପ୍ରତିଛବି ବିଶ୍ଳେଷଣ କରୁଛି...",
        "result_disease": "✅ ରୋଗ ଚିହ୍ନଟ ହୋଇଛି:",
        "result_plant": "🍃 ଉଦ୍ଭିଦ ପ୍ରକାର:",
        "select_page": "ପୃଷ୍ଠା ଚୟନ କରନ୍ତୁ",
        "home": "ଘର",
        "about": "ବିଷୟରେ",
        "disease_recognition": "ରୋଗ ଚିହ୍ନଟ"
    }
}

# Language display names
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "हिन्दी (Hindi)",
    "bn": "বাংলা (Bengali)",
    "ta": "தமிழ் (Tamil)",
    "te": "తెలుగు (Telugu)",
    "mr": "मराठी (Marathi)",
    "gu": "ગુજરાતી (Gujarati)",
    "kn": "ಕನ್ನಡ (Kannada)",
    "ml": "മലയാളം (Malayalam)",
    "pa": "ਪੰਜਾਬੀ (Punjabi)",
    "or": "ଓଡ଼ିଆ (Odia)"
}

# Initialize language
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# Apply custom CSS
def apply_css():
    st.markdown("""
    <style>
        :root {
            --primary: #4CAF50;
            --secondary: #6dcf6d;
            --accent: #FFD700;
            --dark: #1a1a2e;
            --light: #f8f9fa;
            --success: #28a745;
            --info: #17a2b8;
            --warning: #ffc107;
            --danger: #dc3545;
        }
        
        /* Full page gradient background with animated particles */
        .stApp {
            background: linear-gradient(135deg, #f5f9f5 0%, #e3f2e3 100%);
            background-attachment: fixed;
            min-height: 100vh;
            position: relative;
            overflow-x: hidden;
        }
        
        /* Animated floating particles */
        .particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            pointer-events: none;
        }
        
        .particle {
            position: absolute;
            background: radial-gradient(circle, var(--primary) 0%, rgba(76, 175, 80, 0) 70%);
            border-radius: 50%;
            opacity: 0.3;
            animation: floatParticle linear infinite;
        }
        
        @keyframes floatParticle {
            0% {
                transform: translateY(0) translateX(0);
                opacity: 0;
            }
            10% {
                opacity: 0.3;
            }
            90% {
                opacity: 0.3;
            }
            100% {
                transform: translateY(-100vh) translateX(20px);
                opacity: 0;
            }
        }
        
        /* Vibrant header with animation */
        .header {
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            padding: 1.5rem;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            position: relative;
            overflow: hidden;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(
                to bottom right,
                rgba(255, 255, 255, 0.3) 0%,
                rgba(255, 255, 255, 0) 60%
            );
            transform: rotate(30deg);
            animation: shine 6s infinite;
        }
        
        @keyframes shine {
            0% { left: -100%; }
            20% { left: 100%; }
            100% { left: 100%; }
        }
        
        /* Glowing title with gradient text */
        .title {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(to right, #4CAF50, #2E7D32, #FFD700);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin: 1rem 0 2rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            animation: glow 2s ease-in-out infinite alternate;
            position: relative;
            display: inline-block;
            padding: 0 1rem;
        }
        
        .title::after {
            content: '';
            position: absolute;
            bottom: -5px;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(to right, #4CAF50, #FFD700);
            border-radius: 3px;
            transform: scaleX(0);
            transform-origin: left;
            transition: transform 0.3s ease;
        }
        
        .title:hover::after {
            transform: scaleX(1);
        }
        
        @keyframes glow {
            from {
                text-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
            }
            to {
                text-shadow: 0 0 15px rgba(76, 175, 80, 0.8), 0 0 20px rgba(255, 215, 0, 0.6);
            }
        }
        
        /* Premium 3D cards with hover effects */
        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 2.5rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 2.5rem;
            border: none;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255,255,255,0.3);
        }
        
        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 5px;
            background: linear-gradient(to right, var(--primary), var(--accent));
        }
        
        .card:hover {
            transform: translateY(-10px) scale(1.02);
            box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        }
        
        .card:hover::before {
            animation: rainbow 2s linear infinite;
        }
        
        @keyframes rainbow {
            0% { background-position: 0% 50%; }
            100% { background-position: 100% 50%; }
        }
        
        /* Animated buttons with ripple effect */
        .stButton>button {
            background: linear-gradient(145deg, var(--primary), var(--secondary));
            color: white;
            font-size: 1.1rem;
            font-weight: 600;
            padding: 0.9rem 2rem;
            border-radius: 50px;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4);
            border: none;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            z-index: 1;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(76, 175, 80, 0.6);
            background: linear-gradient(145deg, var(--secondary), var(--primary));
        }
        
        .stButton>button:active {
            transform: translateY(1px);
        }
        
        .stButton>button::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 5px;
            height: 5px;
            background: rgba(255, 255, 255, 0.5);
            opacity: 0;
            border-radius: 100%;
            transform: scale(1, 1) translate(-50%);
            transform-origin: 50% 50%;
        }
        
        .stButton>button:focus:not(:active)::after {
            animation: ripple 1s ease-out;
        }
        
        @keyframes ripple {
            0% {
                transform: scale(0, 0);
                opacity: 0.5;
            }
            20% {
                transform: scale(25, 25);
                opacity: 0.3;
            }
            100% {
                opacity: 0;
                transform: scale(40, 40);
            }
        }
        
        /* Enhanced sidebar with glass morphism */
        .sidebar .sidebar-content {
            background: rgba(255, 255, 255, 0.9) !important;
            backdrop-filter: blur(10px);
            box-shadow: 5px 0 15px rgba(0,0,0,0.05);
            border-right: 1px solid rgba(255,255,255,0.3);
        }
        
        /* Language selector styling */
        .language-selector {
            margin-bottom: 1.5rem;
            padding: 1rem;
            background: rgba(255,255,255,0.8);
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        /* File uploader styling */
        .stFileUploader>div>div>div>div {
            border: 2px dashed var(--primary) !important;
            border-radius: 15px !important;
            padding: 2rem !important;
            background: rgba(255,255,255,0.7) !important;
            transition: all 0.3s ease !important;
        }
        
        .stFileUploader>div>div>div>div:hover {
            border-color: var(--accent) !important;
            background: rgba(255,255,255,0.9) !important;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .title {
                font-size: 2.2rem;
            }
            .card {
                padding: 1.5rem;
            }
            .header {
                padding: 1rem;
            }
        }
        
        /* Floating leaves animation */
        .leaf {
            position: fixed;
            background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="%234CAF50"><path d="M17 8C8 10 5.9 16.8 3 22c5-3 10-5 14-7-3-2-5-5-5-7 0-3 3-6 5-6s5 3 5 6c0 2-2 5-5 7 4 2 9 4 14 7-2.9-5.2-5-12-14-14-1.5 2.5-1.5 5.5-1 8z"/></svg>');
            background-size: cover;
            opacity: 0.6;
            animation: falling linear infinite;
            z-index: -1;
            pointer-events: none;
        }
        
        @keyframes falling {
            0% {
                transform: translate(0, -10vh) rotate(0deg);
                opacity: 0;
            }
            10% {
                opacity: 0.6;
            }
            90% {
                opacity: 0.6;
            }
            100% {
                transform: translate(calc(var(--random-x) * 100vw), 100vh) rotate(360deg);
                opacity: 0;
            }
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(0,0,0,0.05);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(var(--primary), var(--secondary));
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent);
        }
    </style>
    
    <!-- Floating particles animation -->
    <script>
        function createParticles() {
            const container = document.createElement('div');
            container.className = 'particles';
            document.body.appendChild(container);
            
            const particleCount = 20;
            for (let i = 0; i < particleCount; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                
                const size = Math.random() * 100 + 50;
                const duration = Math.random() * 20 + 10;
                const delay = Math.random() * 10;
                const left = Math.random() * 100;
                
                particle.style.width = `${size}px`;
                particle.style.height = `${size}px`;
                particle.style.left = `${left}%`;
                particle.style.top = `${Math.random() * 100 + 100}%`;
                particle.style.animationDuration = `${duration}s`;
                particle.style.animationDelay = `${delay}s`;
                
                container.appendChild(particle);
            }
        }
        
        window.addEventListener('load', createParticles);
    </script>
    
    <!-- Floating leaves animation -->
    <script>
        function createLeaves() {
            const container = document.createElement('div');
            container.className = 'leaves-container';
            document.body.appendChild(container);
            
            const leafCount = 15;
            for (let i = 0; i < leafCount; i++) {
                const leaf = document.createElement('div');
                leaf.className = 'leaf';
                
                const size = Math.random() * 20 + 10;
                const duration = Math.random() * 10 + 10;
                const delay = Math.random() * 10;
                const left = Math.random() * 100;
                
                leaf.style.width = `${size}px`;
                leaf.style.height = `${size}px`;
                leaf.style.left = `${left}%`;
                leaf.style.animationDuration = `${duration}s`;
                leaf.style.animationDelay = `${delay}s`;
                leaf.style.setProperty('--random-x', Math.random() * 0.4 - 0.2);
                
                container.appendChild(leaf);
            }
        }
        
        window.addEventListener('load', createLeaves);
    </script>
    """, unsafe_allow_html=True)



# Language selector
def language_selector():
    st.sidebar.markdown("### 🌐 Select Language")
    lang = st.sidebar.selectbox(
        "", 
        options=list(TRANSLATIONS.keys()),
        index=list(TRANSLATIONS.keys()).index(st.session_state.language),
        format_func=lambda x: LANGUAGE_NAMES[x]
    )
    st.session_state.language = lang

# Sidebar Navigation
language_selector()
st.sidebar.title("🌿 Plant Disease Recognition")
t = TRANSLATIONS[st.session_state.language]
app_mode = st.sidebar.radio(
    t["select_page"], 
    [f"🏠 {t['home']}", f"📖 {t['about']}", f"🦠 {t['disease_recognition']}"]
)

# Get current language translations
t = TRANSLATIONS[st.session_state.language]

# Home Page
if app_mode == f"🏠 {t['home']}":
    st.markdown(f"<div class='title'>{t['title']}</div>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image("home_page.jpeg", caption=t['home_title'], use_column_width=True)
        st.markdown(f"""
        ## {t['welcome']}
        {t['description']} 🌍🌿
        
        {t['how_it_works']}
        {t['step1']}
        {t['step2']}
        {t['step3']} 🚀
        
        {t['start']}
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# About Page
elif app_mode == f"📖 {t['about']}":
    st.title(t['about_title'])
    
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"""
        {t['developed_by']}
        {t['team']}
        
        *VIT Open Source Initiative.*
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == f"🦠 {t['disease_recognition']}":
    st.markdown(f"<div class='title'>{t['recognition_title']}</div>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        plant_image = st.file_uploader(t['upload_text'], type=["jpg", "png", "jpeg"])

        if plant_image is not None:
            st.image(plant_image, caption="Uploaded Image", use_column_width=True)
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            opencv_image = cv2.resize(opencv_image, (256, 256))
            opencv_image.shape = (1, 256, 256, 3)

        if st.button(t['predict_button']):
            with st.spinner(t['analyzing']):
                time.sleep(2)  # Simulated delay
                Y_pred = model.predict(opencv_image)
                result = CLASS_NAMES[np.argmax(Y_pred)]

            st.success(f"{t['result_disease']} **{result.split('-')[1]}**")
            st.info(f"{t['result_plant']} **{result.split('-')[0]}**")
            
        st.markdown("</div>", unsafe_allow_html=True)