import numpy as np
import pickle
from transformers import GPT2Tokenizer, TFGPT2Model

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
    Optimistic_Estimate = int(args[0])
    Most_Likely_Estimate = int(args[1])
    Pessimistic_Estimate = int(args[2])


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


def load_and_fit_encoder(file_path, train_data):

    new_list = []
    new_list.append(train_data) 

    with open(file_path, 'rb') as file:
        loaded_encoder = pickle.load(file)

    print(type(loaded_encoder))
    
    train_data_reshaped = np.array(new_list).reshape(-1, 1)
    fitted_encoder = loaded_encoder.transform(train_data_reshaped)
    
    return fitted_encoder

def encodings(*params):
    task_name = params[0]
    task_descrption = params[1]
    task_label = params[2]
    original_estimate = int(params[3])

    Weighted_average_int = params[4]
    Risk_factor_int = params[5]
    Variance = params[6]
    Final_estimate = params[7]

    #Calcuate encoding 
    tl_enc = load_and_fit_encoder('AI/onehotencodings/Task_label/one_hot_task_Label.pkl', task_label)
    pe_enc = load_and_fit_encoder('AI/onehotencodings/Planned_estimate/one_hot_Planned_estimate.pkl', original_estimate)
    fe_enc = load_and_fit_encoder('AI/onehotencodings/Final_estimate/one_hot_final_estimate.pkl',  Final_estimate)
    wa_enc = load_and_fit_encoder('AI/onehotencodings/Weighted_avg_int/one_hot_Weighted_avg_int.pkl',  Weighted_average_int)
    rfi_enc = load_and_fit_encoder('AI/onehotencodings/Risk_factor_int/one_hot_Risk_factor_int.pkl',  Risk_factor_int)
    v_enc = load_and_fit_encoder('AI/onehotencodings/variance_int/one_hot_Complexity_factor_int.pkl', Variance)
    
    gpt_model = GPT()

    tn_testdata = gpt_model.vectors(task_name)

    td_testdata = gpt_model.vectors(task_descrption)

    task_label_list = []
    task_label_list.append(task_label) 

    original_estimate_list = []
    original_estimate_list.append(original_estimate) 

    Final_estimate_list = []
    Final_estimate_list.append(Final_estimate) 


    Weighted_average_int_list = []
    Weighted_average_int_list.append(Weighted_average_int) 

    Risk_factor_int_list = []
    Risk_factor_int_list.append(task_label) 

    Variance_list = []
    Variance_list.append(Variance) 


    # tl_testdata = task_label.to_numpy().reshape(-1,1)
    tl_testdata = np.array(task_label_list).reshape(-1, 1)
    tl_testdata = tl_enc.transform(tl_testdata)

    pe_testdata = original_estimate.to_numpy().reshape(-1,1)
    # pe_testdata = np.array(original_estimate_list).reshape(-1,1)
    pe_testdata = pe_enc.transform(pe_testdata)

    fe_testdata = Final_estimate.to_numpy().reshape(-1,1)
    # fe_testdata = np.array(Final_estimate_list).reshape(-1,1)
    fe_testdata = fe_enc.transform(fe_testdata)

    wa_testdata = Weighted_average_int.to_numpy().reshape(-1,1)
    # wa_testdata = np.array(Weighted_average_int_list).reshape(-1,1)
    wa_testdata = wa_enc.transform(wa_testdata)

    rfi_testdata = Risk_factor_int.to_numpy().reshape(-1,1)
    # rfi_testdata = np.array(Risk_factor_int_list).reshape(-1,1)
    rfi_testdata = rfi_enc.transform(rfi_testdata)

    v_testdata = Variance.to_numpy().reshape(-1,1)
    # v_testdata = np.array(Variance_list).reshape(-1,1)
    v_testdata = v_enc.transform(v_testdata)

    return[tn_testdata,td_testdata,tl_testdata,pe_testdata,fe_testdata,wa_testdata,rfi_testdata,v_testdata]


def predictions(EST_Model,*var):
    tn_testdata = var[0] 
    td_testdata = var[1] 
    tl_testdata = var[2] 
    pe_testdata = var[3] 
    fe_testdata = var[4] 
    wa_testdata = var[5] 
    rfi_testdata = var[6] 
    v_testdata = var[7]

    Ai_estimate = EST_Model.predict([tn_testdata, td_testdata, tl_testdata, pe_testdata, fe_testdata, wa_testdata, rfi_testdata, v_testdata])
    return Ai_estimate

