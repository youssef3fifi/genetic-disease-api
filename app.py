from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# ==========================================
# 1. تحميل الموديل والبيانات عند تشغيل السيرفر
# ==========================================
try:
    print("Loading model_data.pkl...")
    data = joblib.load('model_data.pkl')
    
    rf_model_disease = data["model_disease"]
    model_risk = data["model_risk"]
    feature_encoders = data["feature_encoders"]
    target_encoder_disease = data["target_encoder_disease"]
    target_encoder_risk = data["target_encoder_risk"]
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # لن نوقف التطبيق لكي تظهر رسالة الخطأ في الـ Logs إذا حدثت مشكلة

# ==========================================
# 2. دالة مساعدة للتعامل مع القيم الجديدة (Unseen Labels)
# ==========================================
def safe_transform(encoder, value):
    try:
        # تحويل القيمة لنص لضمان التوافق
        value = str(value)
        # جلب الكلاسات الموجودة في الـ Encoder كـ نصوص
        classes = [str(c) for c in encoder.classes_]
        
        if value in classes:
            return encoder.transform([value])[0]
        else:
            # إذا كانت القيمة غريبة، نستخدم أول قيمة معروفة كبديل (أو قيمة شائعة)
            # هذا يمنع الموديل من الانهيار
            return encoder.transform([encoder.classes_[0]])[0]
    except:
        return 0

# ==========================================
# 3. المسارات (Routes)
# ==========================================

# --- الصفحة الرئيسية (عشان لما تفتح اللينك في المتصفح متلاقيش Error 404) ---
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "online",
        "message": "Welcome to Genetic Disease Prediction API! Use POST /predict_csv to send files.",
        "python_version": "Check Logs for details"
    })

# --- مسار استقبال ملف CSV والتوقع ---
@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        # 1. التأكد من وجود ملف في الطلب
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"})

        file = request.files['file']
        
        # 2. قراءة ملف CSV
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({"status": "error", "message": f"Could not read CSV file: {str(e)}"})
        
        # الأعمدة المطلوبة بالترتيب
        required_columns = [
            'Father_Gene', 'Father_Variant', 'Father_Pathogenicity', 'Father_Inheritance',
            'Mother_Gene', 'Mother_Variant', 'Mother_Pathogenicity', 'Mother_Inheritance',
            'Child_Genotype'
        ]
        
        # التحقق من وجود الأعمدة
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
             return jsonify({"status": "error", "message": f"Missing columns: {missing_cols}"})

        # أخذ نسخة من البيانات للمعالجة
        df_processed = df[required_columns].copy()

        # 3. معالجة البيانات (Encoding)
        for col in df_processed.columns:
            if col in feature_encoders:
                df_processed[col] = df_processed[col].apply(lambda x: safe_transform(feature_encoders[col], x))

        # 4. التوقع (Batch Prediction)
        disease_pred_ids = rf_model_disease.predict(df_processed)
        risk_pred_ids = model_risk.predict(df_processed)

        # 5. فك التشفير (Decoding Results)
        disease_results = target_encoder_disease.inverse_transform(disease_pred_ids)
        risk_results = target_encoder_risk.inverse_transform(risk_pred_ids)

        # 6. تجهيز الرد JSON
        results = []
        for i in range(len(df)):
            results.append({
                "row_index": i,
                "disease_prediction": disease_results[i],
                "risk_prediction": risk_results[i],
                # إرجاع بعض البيانات الأصلية للمساعدة في العرض (اختياري)
                "child_genotype": int(df.iloc[i]['Child_Genotype']) if 'Child_Genotype' in df.columns else 0
            })

        return jsonify({
            "status": "success",
            "total_rows": len(df),
            "predictions": results
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    # تشغيل السيرفر (Render بيستخدم gunicorn بس ده عشان لو شغلته لوكال)
    app.run(host='0.0.0.0', port=10000, debug=True)
