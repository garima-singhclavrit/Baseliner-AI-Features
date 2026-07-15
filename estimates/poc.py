import numpy as np
import pickle
from transformers import GPT2Tokenizer, TFGPT2Model
import tensorflow as tf

class GPT:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = TFGPT2Model.from_pretrained('gpt2')


    def inp_vector(self, inp):
        if type(inp) != 'str':
            inp = "NA"
 
        encoded_inp = self.tokenizer(inp, return_tensors='tf')
        out = self.model(encoded_inp)
        vector = out[0].numpy()[0]
 
        feat_vectors = []
        feat_vectors.append(np.mean(vector, axis=0))
        return np.array(feat_vectors)

def preprocessing(df):

    Optimistic_Estimate = df['Optimistic_Estimate']
    Most_Likely_Estimate = df['Most_Likely_Estimate']
    Pessimistic_Estimate = df['Pessimistic_Estimate']

    Weighted_average = round(((Optimistic_Estimate + (4*Most_Likely_Estimate) + Pessimistic_Estimate)/6), 2)
    Weighted_average_int = int(Weighted_average*100)
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
    Final_estimate = int(Final_estimate)
    Risk_factor_int = int(Risk_factor*100)

    return [Weighted_average_int,Risk_factor_int,Variance,Final_estimate]


def load_and_fit_encoder(file_path, test_data):

    new_list = []
    new_list.append(test_data) 

    with open(file_path, 'rb') as file:      
        loaded_encoder = pickle.load(file)
    
    test_data_reshaped = np.array(new_list).reshape(-1, 1)
    one_hot_encoded_vector = loaded_encoder.transform(test_data_reshaped)
    
    return one_hot_encoded_vector

def encodings(*params):
    task_name = params[0]
    task_descrption = params[1]
    task_label = params[2]
    planned_estimate = int(params[3])

    Weighted_average_int = params[4]
    Risk_factor_int = params[5]
    Variance = params[6]
    Final_estimate = params[7]

    gpt_model = GPT()
    # GPT-2 embedded vectors
    tn_testdata = gpt_model.inp_vector(task_name)
    td_testdata = gpt_model.inp_vector(task_descrption)

    # One hot encoded vectors 
    tl_testdata = load_and_fit_encoder('AI/onehotencodings/Task_label/one_hot_task_Label.pkl', task_label)
    pe_testdata = load_and_fit_encoder('AI/onehotencodings/Planned_estimate/one_hot_Planned_estimate.pkl', planned_estimate)
    fe_testdata = load_and_fit_encoder('AI/onehotencodings/Final_estimate/one_hot_final_estimate.pkl',  Final_estimate)
    wa_testdata = load_and_fit_encoder('AI/onehotencodings/Weighted_avg_int/one_hot_Weighted_avg_int.pkl',  Weighted_average_int)
    rfi_testdata = load_and_fit_encoder('AI/onehotencodings/Risk_factor_int/one_hot_Risk_factor_int.pkl',  Risk_factor_int)
    v_testdata = load_and_fit_encoder('AI/onehotencodings/variance_int/one_hot_Complexity_factor_int.pkl', Variance)
    

    return [tn_testdata, td_testdata, tl_testdata, pe_testdata, fe_testdata, wa_testdata, rfi_testdata, v_testdata]

def predictions(EST_Model, *var):
    tn_testdata = var[0] 
    td_testdata = var[1] 
    tl_testdata = var[2] 
    pe_testdata = var[3] 
    fe_testdata = var[4] 
    wa_testdata = var[5] 
    rfi_testdata = var[6] 
    v_testdata = var[7]

    predictions = EST_Model.predict([tn_testdata, td_testdata, tl_testdata, pe_testdata, fe_testdata, wa_testdata, rfi_testdata, v_testdata])

    file_path = 'AI/onehotencodings/Actual_estimate/one_hot_Actual_estimate_int.pkl'
    with open(file_path, 'rb') as file:
        ae_enc = pickle.load(file)

    Ai_estimate = ae_enc.inverse_transform(predictions)
    return Ai_estimate
