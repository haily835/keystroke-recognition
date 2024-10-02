from utils.calculate_metrics import calculate_metrics
from sklearn.metrics import classification_report

print("Clf:")

for model in ['HyperGT_add_data', 'HDGCN_add_data', 'CTRGCN_add_data', 'STGCN_add_data']:
    result, _, _ = calculate_metrics(f"./ckpts/{model}/clf/clf_test_results.csv")
    print(f"{model.upper()} & {result} \\\\ \hline")
    # calculate_metrics(f"./ckpts/{model}/det/det_test_results.csv")
    

print("Detection:")

for model in ['HyperGT_add_data', 'HDGCN_add_data', 'CTRGCN_add_data', 'STGCN_add_data']:
    result, _, _ = calculate_metrics(f"./ckpts/{model}/det/det_test_results.csv")
    print(f"{model.upper()} & {result} \\\\ \hline")

result,  y_pred, y_true = calculate_metrics(f"./ckpts/HyperGT_add_data/clf/clf_test_results.csv")
print(classification_report(y_true, y_pred, digits=4))