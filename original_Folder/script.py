
# Task name
tn_testdata = gpt_model.vectors(df_test["Task_name"])
print(type(tn_testdata))
print(tn_testdata.shape)


td_testdata = gpt_model.vectors(df_test["Task_description"])
print(type(td_testdata))
print(td_testdata.shape)


tl_testdata = gpt_model.vectors(df_test["Task_label"])
print(type(tl_testdata))
print(tl_testdata.shape)

pe_testdata = gpt_model.vectors(df_test["Planned_estimate"])
print(type(pe_testdata))
print(pe_testdata.shape)


