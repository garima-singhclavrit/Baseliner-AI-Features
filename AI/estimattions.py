import numpy as np
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2Model
import pickle
import time

class GPT:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = TFGPT2Model.from_pretrained('gpt2')

    def inp_vector(self, inp):
        encoded_inp = self.tokenizer(inp, return_tensors='tf')
        out = self.model(encoded_inp)
        vector = out[0].numpy()[0]
        return np.mean(vector, axis=0)

    def vectors(self, inps):
        feat_vectors = []
        for i in inps:
            if type(i) != 'str':
                i = "NA"
            feat_vectors.append(self.inp_vector(i))
        return np.array(feat_vectors)


def preprocessing(*args):
    Optimistic_Estimate = args[0]
    Most_Likely_Estimate = args[1]
    Pessimistic_Estimate = args[2]


    Weighted_average = round(((Optimistic_Estimate + (4*Most_Likely_Estimate) + Pessimistic_Estimate)/6), 2)
    Weighted_average_int = (Weighted_average*100).astype(int)
    Standard_deviation = round(((Pessimistic_Estimate - Optimistic_Estimate)/6), 2)
    
    Risk_factor = round((Standard_deviation/Weighted_average), 2)
    Standard_confidence = Weighted_average
    High_confidence = round((Weighted_average + (0.53 * Standard_deviation)), 2)
    Higher_confidence = round((Weighted_average + (1.28 * Standard_deviation)), 2)
    Highest_confidence = round((Weighted_average + (2.33 * Standard_deviation)), 2)
    Final_estimate = round(Highest_confidence, 1)
    Variance = round(((6*Pessimistic_Estimate - Optimistic_Estimate)**2), 2)

    Final_estimate = np.where((Risk_factor >= 0) & (Risk_factor <= 0.05), Standard_confidence, Final_estimate)
    Final_estimate = np.where((Risk_factor > 0.05) & (Risk_factor <= 0.1), High_confidence, Final_estimate)
    Final_estimate = np.where((Risk_factor > 0.1) & (Risk_factor <= 0.15), Higher_confidence, Final_estimate)
    Final_estimate = Final_estimate.astype(int)
    Risk_factor_int = (Risk_factor*100).astype(int)



    return [Weighted_average_int,Risk_factor_int,Variance,Final_estimate]



def encodings(*argument):

    task_name = argument[0]
    task_descrption = argument[1]
    label = argument[2]
    planned_estimate = argument[3]
    Actual_estimate = argument[4]

    Weighted_average_int = argument[5]
    Risk_factor_int = argument[6]
    Variance = argument[7]
    Final_estimate = argument[8]


    model_start = time.time()
    file_path = 'AI/onehotencodings/Task_label/ohe_tl_train.pkl'
    with open(file_path, 'rb') as file:    
        tl_enc = pickle.load(file)
    tl_train = label.to_numpy().reshape(-1,1)
    tl_enc.fit(tl_train)

    file_path = 'AI/onehotencodings/Planned_estimate/ohe_pe_train.pkl'
    with open(file_path, 'rb') as file:    
        pe_enc = pickle.load(file)

    pe_train = original_estimate.to_numpy().reshape(-1,1)
    pe_enc.fit(pe_train)


    file_path = 'AI/onehotencodings/Final_estimate/ohe_fe_train.pkl'
    with open(file_path, 'rb') as file:    
        fe_enc = pickle.load(file)

    fe_train = Final_estimate.to_numpy().reshape(-1,1)
    fe_enc.fit(fe_train)


    file_path = 'AI/onehotencodings/Weighted_avg_int/ohe_wa_train.pkl'
    with open(file_path, 'rb') as file:    
        wa_enc = pickle.load(file)

    wa_train = Weighted_average_int.to_numpy().reshape(-1,1)
    wa_enc.fit(wa_train)


    file_path = 'AI/onehotencodings/Risk_factor_int/ohe_rfi_train.pkl'
    with open(file_path, 'rb') as file:    
        rfi_enc = pickle.load(file)

    rfi_train = Risk_factor_int.to_numpy().reshape(-1,1)
    rfi_enc.fit(rfi_train)


    file_path = 'AI/onehotencodings/variance_int/ohe_v_test.pkl'
    with open(file_path, 'rb') as file:    
        v_enc = pickle.load(file)
    v_train = Variance.to_numpy().reshape(-1,1)
    v_enc.fit(v_train)
    
    model_end = time.time()
    print(f"Model loading time is : {model_end- model_start}")

    # ae_enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
    # ae_train = Actual_estimate.to_numpy().reshape(-1,1)
    # ae_enc.fit(ae_train)

   
    gpt_model = GPT()

    tn_testdata = gpt_model.vectors(task_name)
    print(type(tn_testdata))
    print(tn_testdata.shape)

    td_testdata = gpt_model.vectors(task_descrption)
    print(type(td_testdata))
    print(td_testdata.shape)

    tl_testdata = label.to_numpy().reshape(-1,1)
    tl_testdata = tl_enc.transform(tl_testdata)
    print(tl_testdata.shape)

    pe_testdata = original_estimate.to_numpy().reshape(-1,1)
    pe_testdata = pe_enc.transform(pe_testdata)
    print(pe_testdata.shape)

    fe_testdata = Final_estimate.to_numpy().reshape(-1,1)
    fe_testdata = fe_enc.transform(fe_testdata)
    print(fe_testdata.shape)

    # wa_testdata = Weighted_average_.to_numpy().reshape(-1,1)
    wa_testdata = Weighted_average_int.to_numpy().reshape(-1,1)
    wa_testdata = wa_enc.transform(wa_testdata)
    print(wa_testdata.shape)

    rfi_testdata = Risk_factor_int.to_numpy().reshape(-1,1)
    rfi_testdata = rfi_enc.transform(rfi_testdata)
    print(rfi_testdata.shape)

    v_testdata = Variance.to_numpy().reshape(-1,1)
    print(v_testdata.shape)


    # ae_testdata = Actual_estimate.to_numpy().reshape(-1,1)
    # ae_testdata = ae_enc.transform(ae_testdata)
    # print(ae_testdata.shape)



    return[tn_testdata,td_testdata,tl_testdata,pe_testdata,fe_testdata,wa_testdata,rfi_testdata,v_testdata]




def predictions(*var):
    model7 = tf.keras.models.load_model('drive/MyDrive/Colab Notebooks/Estimates/Model7/model7.h5')
    model7.summary()

    tn_testdata = var[0] 
    td_testdata = var[1] 
    tl_testdata = var[2] 
    pe_testdata = var[3] 
    fe_testdata = var[4] 
    wa_testdata = var[5] 
    rfi_testdata = var[6] 
    v_testdata = var[7]
    ae_enc = var[8]


    predictions = model7.predict([tn_testdata, td_testdata, tl_testdata, pe_testdata, fe_testdata, wa_testdata, rfi_testdata, v_testdata])

    print(predictions.shape)


    y_pred_label = ae_enc.inverse_transform(predictions)

    # print(y_actual_label.shape, y_pred_label.shape)

    Ai_estimate = y_pred_label
    return Ai_estimate



task_name = 'name from DB'
task_descrption = 'description from DB'
label = 'lables from db'
original_estimate = 'estimate from DB'
Actual_estimate = 'estimate from DB'
sprint_number = 'From DB'
priority = 'From DB'


Optimistic_Estimate = request.data.get('Optimistic_estimate')
Most_Likely_Estimate = request.data.get('Most_likely_estimate')
Pessimistic_Estimate = request.data.get('Pessimistic_estimate')

params = [Optimistic_Estimate,Most_Likely_Estimate,Pessimistic_Estimate,Actual_estimate]

var = preprocessing(*params)

params1 = [task_name,task_descrption,label,original_estimate,Actual_estimate]
params2 = var.extend(params1)
print(params2)

params3 = encodings(*params2)

ai_estimate = predictions(params3)

print(ai_estimate)



