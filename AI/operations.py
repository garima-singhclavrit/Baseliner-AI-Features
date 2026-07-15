import numpy as np
import pickle
import tensorflow as tf
import time

def preprocessing(df):
    """
    Preprocesses input DataFrame to calculate various estimates and metrics.

    This function takes a DataFrame containing task estimates and calculates 
    the weighted average, standard deviation, risk factor, final estimate, 
    and other metrics based on Optimistic, Most Likely, and Pessimistic estimates. 
    The final estimate is adjusted based on the calculated risk factor.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing task information with the following columns:
        - 'Optimistic_Estimate': The optimistic estimate for task completion.
        - 'Most_Likely_Estimate': The most likely estimate for task completion.
        - 'Pessimistic_Estimate': The pessimistic estimate for task completion.

    Returns
    -------
    list
        A list containing the following calculated values:
        - Weighted_average_int (int): Weighted average multiplied by 100, as an integer.
        - Risk_factor_int (int): Risk factor percentage, multiplied by 100 and rounded to an integer.
        - Variance (float): Calculated variance based on estimates.
        - Final_estimate (int): The final estimate adjusted based on the risk factor.
    
    Raises
    ------
    ValueError
        If any of the estimate values are non-numeric.
    """
        
    Optimistic_Estimate = int(df._get_value(0, 'Optimistic_Estimate')) 
    Most_Likely_Estimate = int(df._get_value(0, 'Most_Likely_Estimate'))
    Pessimistic_Estimate = int(df._get_value(0, 'Pessimistic_Estimate'))

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

def encodings(df,gpt_model,*params):
    task_name = df._get_value(0, 'task_name')
    task_description = df._get_value(0, 'task_description')
    task_label = df._get_value(0, 'task_label')
    planned_estimate = int(df._get_value(0, 'planned_estimate'))

    Weighted_average_int = params[0]
    Risk_factor_int = params[1]
    Variance = params[2]
    Final_estimate = params[3]
    
    model_start = time.time()
    #gpt_model = GPT()
    # GPT-2 embedded vectors
    tn_testdata = gpt_model.inp_vector(task_name)
    td_testdata = gpt_model.inp_vector(task_description)
    model_end = time.time()
    print(f"GPT Model loading time is : {model_end- model_start}")

    # One hot encoded vectors 
    tl_testdata = load_and_fit_encoder('AI/onehotencodings/Task_label/one_hot_task_Label.pkl', task_label)
    pe_testdata = load_and_fit_encoder('AI/onehotencodings/Planned_estimate/one_hot_Planned_estimate.pkl', planned_estimate)
    fe_testdata = load_and_fit_encoder('AI/onehotencodings/Final_estimate/one_hot_final_estimate.pkl',  Final_estimate)
    wa_testdata = load_and_fit_encoder('AI/onehotencodings/Weighted_avg_int/one_hot_Weighted_avg_int.pkl',  Weighted_average_int)
    rfi_testdata = load_and_fit_encoder('AI/onehotencodings/Risk_factor_int/one_hot_Risk_factor_int.pkl',  Risk_factor_int)
    v_testdata = load_and_fit_encoder('AI/onehotencodings/variance_int/one_hot_Complexity_factor_int.pkl', Variance)
    

    return [tn_testdata, td_testdata, tl_testdata, pe_testdata, fe_testdata, wa_testdata, rfi_testdata, v_testdata]

def predictions(EST_Model, *var):
    """
    Generates AI-based predictions for task estimates using a pre-trained model.

    This function takes multiple test data inputs and uses a pre-trained model 
    to predict task estimates. The predicted output is then decoded using an 
    inverse transform to obtain the final AI estimate.

    Parameters
    ----------
    EST_Model : object
        A pre-trained model used for making predictions on the input data.
    *var : tuple
        A tuple containing the following test data in order:
        - tn_testdata : Test data for task name encoding.
        - td_testdata : Test data for task description encoding.
        - tl_testdata : Test data for task label encoding.
        - pe_testdata : Test data for planned estimate.
        - fe_testdata : Test data for final estimate.
        - wa_testdata : Test data for weighted average.
        - rfi_testdata : Test data for risk factor index.
        - v_testdata : Test data for variance.

    Returns
    -------
    np.ndarray
        The AI estimate obtained by decoding the predicted values.
    
    Raises
    ------
    FileNotFoundError
        If the specified encoding file is not found.
    PickleError
        If there is an error during loading the encoding file.
    """
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
