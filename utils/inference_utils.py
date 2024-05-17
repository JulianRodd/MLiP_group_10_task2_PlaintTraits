import tqdm
import torch 
from pandas import DataFrame

from generics import Generics


def create_submission(model, dataset, scaler, log_features=Generics.TARGET_COLUMNS, device='cuda'):
    '''
    Creates a file "submission.csv" in cwd. 
    
    Takes: 
        model: model to do inference on 
        dataset: test dataset like torch.dataset
        scaler: like sklearn scaler, needs to be fitted
        log_features (optional): by default log features are all target columns
    Returns: 
        None 
    '''
    
    SUBMISSION_ROWS = []
    model.eval()

    for X_sample_test, test_id in tqdm(dataset):
        with torch.no_grad():
            y_pred = model(X_sample_test.unsqueeze(0).to(device)).detach().cpu().numpy()
        
        y_pred = scaler.inverse_transform(y_pred).squeeze()
        row = {'id': test_id}
        
        for k, v in zip(Generics.TARGET_COLUMNS, y_pred):
            if k in log_features:
                row[k.replace('_mean', '')] = 10 ** v
            else:
                row[k.replace('_mean', '')] = v

        SUBMISSION_ROWS.append(row)
        
    submission_df = DataFrame(SUBMISSION_ROWS)
    submission_df.to_csv('submission.csv', index=False)
    print("Ready to submit!")