from flask import Flask, request, jsonify
from flask import Markup
from flask_cors import CORS, cross_origin
from keras import backend as K

import os
import json
import pickle
import numpy as np
from scipy import stats
import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import string
import pandas as pd
import glob
import scipy.sparse as sparse
import keras
import cv2
from pymongo import MongoClient
from pprint import pprint
# from keras_efficientnets import custom_object
from keras.utils.generic_utils import get_custom_objects
from keras_efficientnets import EfficientNetB2
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

import werkzeug

app = Flask(__name__)

cors = CORS(app)

@app.route("/")
def home_view():
        return "<h1>Flask API for Plant Disease</h1>"


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add("Access-Control-Allow-Headers",
                         "Origin, X-Requested-With, Content-Type, Accept")
    response.headers.add('Access-Control-Allow-Methods',
                         'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response


# Loading all Crop Recommendation Models
crop_xgb_pipeline = pickle.load(
    open("./models/crop_recommendation/xgb_pipeline.pkl", "rb")
)
crop_rf_pipeline = pickle.load(
    open("./models/crop_recommendation/rf_pipeline.pkl", "rb")
)
crop_knn_pipeline = pickle.load(
    open("./models/crop_recommendation/knn_pipeline.pkl", "rb")
)
crop_label_dict = pickle.load(
    open("./models/crop_recommendation/label_dictionary.pkl", "rb")
)


# Loading all Fertilizer Recommendation Models
fertilizer_xgb_pipeline = pickle.load(
    open("./models/fertilizer_recommendation/xgb_pipeline.pkl", "rb")
)
fertilizer_rf_pipeline = pickle.load(
    open("./models/fertilizer_recommendation/rf_pipeline.pkl", "rb"))
fertilizer_svm_pipeline = pickle.load(
    open("./models/fertilizer_recommendation/svm_pipeline.pkl", "rb"))
fertilizer_label_dict = pickle.load(
    open("./models/fertilizer_recommendation/fertname_dict.pkl", "rb"))
soiltype_label_dict = pickle.load(
    open("./models/fertilizer_recommendation/soiltype_dict.pkl", "rb"))
croptype_label_dict = pickle.load(
    open("./models/fertilizer_recommendation/croptype_dict.pkl", "rb"))
crop_label_name_dict = {}
for crop_value in croptype_label_dict:

    crop_label_name_dict[croptype_label_dict[crop_value]] = crop_value
    
    soil_label_dict = {}
    
    for soil_value in soiltype_label_dict:
        soil_label_dict[soiltype_label_dict[soil_value]] = soil_value

    def convert(o):
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError

    def crop_prediction(input_data):
        prediction_data = {"xgb_model_prediction": crop_label_dict[crop_xgb_pipeline.predict(input_data)[0]],
                           "xgb_model_probability": max(crop_xgb_pipeline.predict_proba(input_data)[0]) * 100,
                           "rf_model_prediction": crop_label_dict[crop_rf_pipeline.predict(input_data)[0]],
                           "rf_model_probability": max(crop_rf_pipeline.predict_proba(input_data)[0]) * 100,
                           "knn_model_prediction": crop_label_dict[crop_knn_pipeline.predict(input_data)[0]],
                           "knn_model_probability": max(crop_knn_pipeline.predict_proba(input_data)[0]) * 100, }
        all_predictions = [
            prediction_data["xgb_model_prediction"],
            prediction_data["rf_model_prediction"],
            prediction_data["knn_model_prediction"],
        ]

        all_probs = [
            prediction_data["xgb_model_probability"],
            prediction_data["rf_model_probability"],
            prediction_data["knn_model_probability"],
        ]

        if len(set(all_predictions)) == len(all_predictions):
            prediction_data["final_prediction"] = all_predictions[all_probs.index(
                max(all_probs))]
        else:
            prediction_data["final_prediction"] = stats.mode(all_predictions)[0][0]

        return prediction_data


def fertilizer_prediction(input_data):
    prediction_data = {
        "xgb_model_prediction": fertilizer_label_dict[
            fertilizer_xgb_pipeline.predict(input_data)[0]
        ],
        "xgb_model_probability": max(
            fertilizer_xgb_pipeline.predict_proba(input_data)[0]
        )
        * 100,
        "rf_model_prediction": fertilizer_label_dict[
            fertilizer_rf_pipeline.predict(input_data)[0]
        ],
        "rf_model_probability": max(fertilizer_rf_pipeline.predict_proba(input_data)[0])
        * 100,
        "svm_model_prediction": fertilizer_label_dict[
            fertilizer_svm_pipeline.predict(input_data)[0]
        ],
        "svm_model_probability": max(
            fertilizer_svm_pipeline.predict_proba(input_data)[0]
        )
        * 100,
    }

    all_predictions = [
        prediction_data["xgb_model_prediction"],
        prediction_data["rf_model_prediction"],
        prediction_data["svm_model_prediction"],
    ]

    all_probs = [
        prediction_data["xgb_model_probability"],
        prediction_data["rf_model_probability"],
        prediction_data["svm_model_probability"],
    ]

    if len(set(all_predictions)) == len(all_predictions):
        prediction_data["final_prediction"] = all_predictions[all_probs.index(
            max(all_probs))]
    else:
        prediction_data["final_prediction"] = stats.mode(all_predictions)[0][0]

    return prediction_data


@app.route("/predict_crop", methods=["GET", "POST"])
def predictcrop():
    try:
        if request.method == "POST":
            form_values = request.form.to_dict()
            column_names = ["N", "P", "K", "temperature",
                            "humidity", "ph", "rainfall"]
            input_data = np.asarray([float(form_values[i].strip()) for i in column_names]).reshape(
                1, -1
            )
            prediction_data = crop_prediction(input_data)
            json_obj = json.dumps(prediction_data, default=convert)
            return json_obj
    except:
        return json.dumps({"error": "H??y ki???m tra l???i d??? li???u b???n nh???p"}, default=convert)


@app.route("/predict_fertilizer", methods=["GET", "POST"])
def predictfertilizer():
    try:
        if request.method == "POST":
            form_values = request.form.to_dict()
            column_names = [
                "nitrogen",
                "potassium",
                "phosphorous",
                "soil_type",
                "crop_type",
                "temperature",
                "humidity",
                "moisture",
            ]
            crop_dict = {"ng??": "Maize",
                         "m??a": "Sugarcane",
                         "b??ng": "Cotton",
                         "thu???c l??": "Tobacco",
                         "l??a": "Paddy",
                         "l??a m???ch": "Barley",
                         "l??a m??": "Wheat",
                         "k??": "Millets",
                         "h???t d???u": "Oid seeds",
                         "b???t gi???y": "Pulses",
                         "h???t xay": "Ground nuts"}

            soil_dict = {"?????t c??t pha": "Sandy",
                         "?????t pha s??t": "Loamy",
                         "?????t ??en": "Black",
                         "?????t ?????": "Red",
                         "?????t s??t": "Clayey"}

            form_values["crop_type"] = crop_dict[form_values["crop_type"]]
            form_values["soil_type"] = soil_dict[form_values["soil_type"]]

            for key in form_values:
                form_values[key] = form_values[key].strip()

            form_values["crop_type"] = crop_label_name_dict[form_values["crop_type"]]
            form_values["soil_type"] = soil_label_dict[form_values["soil_type"]]
            input_data = np.asarray([float(form_values[i]) for i in column_names]).reshape(
                1, -1
            )
            prediction_data = fertilizer_prediction(input_data)
            json_obj = json.dumps(prediction_data, default=convert)
            return json_obj
    except:
        return json.dumps({"error": "Please Enter Valid Data"}, default=convert)


data = ['B???nh v???y T??o',
        'B???nh th???i ??en tr??n T??o',
        'B???nh g??? s???t tr??n T??o',
        'T??o kh???e m???nh',
        'B???nh h??o xanh do vi khu???n tr??n Chu???i',
        'B???nh v???t l?? ??en tr??n Chu???i',
        'Chu???i kh???e m???nh',
        'Vi???t qu???t kh???e m???nh',
        'Cherry kh???e m???nh',
        'B???nh ph???n tr???ng tr??n Cherry',
        'B???nh ?????m ??en tr??n Cam Qu??t',
        'B???nh th???i nh??ng tr??n Cam Qu??t',
        'Cam Qu??t kh???e m???nh',
        'B???nh ?????m l?? x??m tr??n Ng??',
        'B???nh g??? s???t tr??n Ng??',
        'Ng?? kh???e m???nh',
        'B???nh ch??y l?? Ng?? B???c',
        'D??a Leo kh???e m???nh',
        'B???nh s????ng mai tr??n D??a Leo',
        'B???nh ?????m n??u c??nh tr??n Thanh Long',
        'B???nh th???i ??en tr??n Nho',
        'B???nh s???i ??en tr??n Nho',
        'Nho kh???e m???nh',
        'B???nh ch??y l?? tr??n Nho',
        'B???nh v??ng l?? g??n xanh tr??n Cam',
        'B???nh ?????m do vi khu???n tr??n ????o',
        '????o kh???e m???nh',
        'B???nh ?????m do vi khu???n tr??n Ti??u',
        'Ti??u kh???e m???nh',
        'B???nh h??o s???m tr??n Khoai T??y',
        'Khoai T??y kh???e m???nh',
        'B???nh h??o mu???n tr??n Khoai T??y',
        'D??u T???m kh???e m???nh',
        'B???nh ?????m v???n tr??n L??a',
        '?????u N??nh kh???e m???nh',
        'B???nh ph???n tr???ng tr??n B??',
        'D??u kh???e m???nh',
        'B???nh l?? ch??y x??m tr??n D??u',
        'B???nh ?????m do vi khu???n tr??n C?? Chua',
        'B???nh h??o s???m tr??n C?? Chua',
        'C?? Chua kh???e m???nh',
        'B???nh h??o mu???n tr??n C?? Chua',
        'B???nh m???c l?? tr??n C?? Chua',
        'B???nh ?????m Septoria tr??n C?? Chua',
        'B???nh ?????m nh???n tr??n C?? Chua',
        'B???nh ?????m ??en tr??n C?? Chua',
        'B???nh kh???m tr??n C?? Chua',
        'B???nh v??ng l?? xo??n tr??n C?? Chua']


@app.route('/predict_image', methods=['GET', 'POST'])
def upload_image():
    # try:
    if request.method == 'POST':
        img_path = ""
        # for web
        form_values = request.form.to_dict()
        for i, j in form_values.items():
            img_path = img_path + j

        # for mobile
        if (img_path == ""):
            imagefile = request.files["image"]
            # Getting file name of the image using werkzeug library
            filename = werkzeug.utils.secure_filename(imagefile.filename)
            # Saving the image in images Directory
            imagefile.save("static/image/" + filename)
            img_path = filename

        if (img_path != ""):
            print('Image received', img_path)
            img = cv2.imread('static/image/' + img_path)
            img = cv2.resize(img/255, (224, 224))
            img = np.reshape(img, [1, 224, 224, 3])
            K.clear_session()
            model = keras.models.load_model('efficientnetb2.h5')

            def get_predict(image):
                img = cv2.imread(image)
                img = cv2.resize(img/255, (224, 224))
                img = np.reshape(img, [1, 224, 224, 3])
                return model.predict(img)

            def get_feature_img(image):
                img = cv2.imread(image)
                img = cv2.resize(img/255, (224, 224))
                img = np.reshape(img, [1, 224, 224, 3])
                inter_model = keras.Model(
                    model.input, model.get_layer(index=2).output)
                return inter_model.predict(img)[0]

            def rc_disease_similarity(image, model):
                feature_image_ = get_feature_img(image)
                doc = name_all_clean[np.argmax(get_predict(image))]
                if (doc not in name_healthy_clean):
                    query_vector1 = tfidf_vectorizer.transform([doc])
                    query_vector = sparse.hstack(
                        (query_vector1, feature_image_))
                    arr = max(cosine_similarity(query_vector, df_new))

                    arr_sort = arr.copy()
                    sort_indices = np.argsort(arr_sort)[::-1]
                #     arr_sort[:] = arr_sort[sort_indices]
                    out = [doc]
                    treat = []
                    i = 0
                    while len(out) < 10:
                        if name_all_clean[sort_indices[i]] not in out and name_all_clean[sort_indices[i]] not in name_healthy_clean:
                            out.append(name_all_clean[sort_indices[i]])
                            print(treatment[i])
                            treat.append(treatment[i])
                        i += 1
                    return [out[1:], treat]
                else:
                    return None

            def clean_document(doc):
                text_clean = "".join(
                    [i.lower() for i in doc if i not in string.punctuation])
                return text_clean
            classes = model.predict(img)
            a = np.argmax(classes)
            # connect database
            client = MongoClient(
                'mongodb+srv://admin:admin@cluster0.iey5z.mongodb.net'
            )
            db = client['plant_disease']
            collection_all = db['name_plant']
            collection_healthy = db['name_plant_healthy']
            collection_treatment = db['plant_dis']
            name_all = []
            name_healthy = []
            treatment = []
            symptom = []
            for i in collection_all.find():
                k = 0
                for j in collection_treatment.find():
                    if i['name'] == j['label_vi']:
                        k = 1
                        symptom.append(j['symptom'])
                        treatment.append(j['treatment'])
                        break
                if k == 0:
                    symptom.append('')
                    treatment.append('')
                name_all.append(i['name'])
            for i in collection_healthy.find():
                name_healthy.append(i['name'])
            # tfdif
            name_all_clean = [clean_document(i) for i in name_all]
            name_healthy_clean = [clean_document(i) for i in name_healthy]

            stopwords_list = ['b???', 'b???i', 'c???', 'c??c', 'c??i', 'c???n', 'c??ng', 'ch???', 'chi???c', 'cho', 'ch???',
                              'ch??a', 'c??', 'c??_th???', 'c???', 'c??ng', 'c??ng', '????', '??ang', '?????', 'do', '????',
                              '???????c', 'g??', 'khi', 'kh??ng', 'l??', 'l???i', 'l??n', 'l??c', 'm??', 'm???i', 'n??y', 'n??n',
                              'n???u', 'ngay', 'nhi???u', 'nh??', 'nh??ng', 'nh???ng', 'n??i', 'n???a', 'ph???i', 'qua', 'ra',
                              'r???ng', 'r???t', 'r???i', 'sau', 's???', 'theo', 'th??', 't???', 't???ng', 'v??', 'v???n', 'v??o', 'v???y', 'v??', 'vi???c', 'v???i']

            tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords_list)
            sparse_matrix = tfidf_vectorizer.fit_transform(name_all_clean)
            index = [i for i in range(1, len(name_all_clean)+1)]
            doc_term_matrix = sparse_matrix.todense()
            df = pd.DataFrame(doc_term_matrix,
                              columns=tfidf_vectorizer.get_feature_names(),
                              index=index)
            feature_img = np.array(np.genfromtxt(
                "img_feature.txt", delimiter=","))
            df_ = pd.DataFrame(feature_img,
                               columns=[i for i in range(12672)],
                               index=[i for i in range(1, 49)])
            df_new = pd.concat([df, df_], axis=1)

            arr = rc_disease_similarity('static/image/' + img_path, model)
            # print(collection_treatment.find()[a])
            rc = ""
            r1 = ""
            r2 = ""
            if arr is not None:
                for i in range(len(arr[0])):
                    rc += "????" + arr[0][i] + "\n" + "???? " + arr[1][i] + "\n"
                # print(len(arr[0]), len(arr[1]))
                print(img_path, "Success")
                # rc = Markup(rc)
                rc = rc.split('\n')
                return {
                    'name': name_all[a],
                    'symptom': symptom[a],
                    'treatment': treatment[a],
                    'rc': rc
                }
            else:
                print(img_path, "Healthy")
                return {'name': name_all[a], 'symptom': "", 'treatment': "", 'rc': ""}
        else:
            print(img_path, "Failed")
            return ""


if __name__ == "__main__":
    # app.run(debug=True)
    app.run()
