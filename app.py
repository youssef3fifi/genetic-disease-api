from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# تحميل الموديل والبيانات
data = joblib.load('model_data.pkl')
rf_model_disease = data["model_disease"]
model_risk = data["model_risk"]
feature_encoders = data["feature_encoders"]
target_encoder_disease = data["target_encoder_disease"]
target_encoder_risk = data["target_encoder_risk"]

# دالة التعامل مع القيم الجديدة
def safe_transform(encoder, value):
    try:
        # تحويل القيمة لنص لضمان المطابقة (لأن الـ CSV قد يقرأ الأرقام كنصوص أو العكس)
        value = str(value) 
        # التأكد من أن القيم في الـ Encoder هي أيضاً نصوص للمقارنة
        classes = [str(c) for c in encoder.classes_]
        
        if value in classes:
            return encoder.transform([value])[0]
        else:
            return encoder.transform([encoder.classes_[0]])[0]
    except:
        return 0

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        # 1. التأكد من وجود ملف في الطلب
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"})

        file = request.files['file']
        
        # 2. قراءة ملف CSV
        df = pd.read_csv(file)
        
        # التأكد من وجود الأعمدة المطلوبة فقط (نفس الأعمدة التي تدرب عليها الموديل)
        required_columns = [
            'Father_Gene', 'Father_Variant', 'Father_Pathogenicity', 'Father_Inheritance',
            'Mother_Gene', 'Mother_Variant', 'Mother_Pathogenicity', 'Mother_Inheritance',
            'Child_Genotype'
        ]
        
        # التحقق من أن الملف يحتوي على الأعمدة المطلوبة
        if not all(col in df.columns for col in required_columns):
             return jsonify({"status": "error", "message": "CSV file missing required columns"})

        # عمل نسخة للبيانات لنستخدمها في التوقع
        df_processed = df[required_columns].copy()

        # 3. معالجة البيانات (Encoding) لكل الأعمدة
        for col in df_processed.columns:
            if col in feature_encoders:
                df_processed[col] = df_processed[col].apply(lambda x: safe_transform(feature_encoders[col], x))

        # 4. التوقع (Batch Prediction)
        disease_pred_ids = rf_model_disease.predict(df_processed)
        risk_pred_ids = model_risk.predict(df_processed)

        # 5. فك التشفير (Decoding Results)
        disease_results = target_encoder_disease.inverse_transform(disease_pred_ids)
        risk_results = target_encoder_risk.inverse_transform(risk_pred_ids)

        # 6. تجهيز الرد (دمج النتائج مع البيانات الأصلية أو إرسال النتائج فقط)
        results = []
        for i in range(len(df)):
            results.append({
                "row_index": i,
                "disease_prediction": disease_results[i],
                "risk_prediction": risk_results[i],
                # يمكنك إضافة بيانات من الملف الأصلي هنا ليعرف المستخدم أي صف تقصد
                "child_genotype": int(df.iloc[i]['Child_Genotype']) 
            })

        return jsonify({
            "status": "success",
            "total_rows": len(df),
            "predictions": results
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)