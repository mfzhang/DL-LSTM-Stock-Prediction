#%% Performance indicators
import math
import numpy as np

#====== INPUTS =======

#   epochs:             number of epochs
# - valid_summary:      interval you make test predictions
# - n_predict_once:     number of steps you continously predict for
# - begin:              begin of test set            
# - end                 end of test set
# - interval            interval of test points - number of enrollings
# - batch_size          size of batch set
# - predictions_over_time  predictions for all the epochs
# - all_mid_data           all data

begin = 11000
end = 12000
interval = 50

def Performance_indicators(epochs,valid_summary,n_predict_once, begin, end, interval, batch_size, predictions_over_time, all_mid_data):
    
    
# Accumulate test over time (for all epochs)
    test_mse_ot = []    # mean squared error
    test_rmse_ot = []   # root mean squared error
    test_mae_ot = []    # mean absolute error
    test_mre_ot = []    # mean relative error
    test_lincor_ot = [] # linear correlation coefficient
    test_maxae_ot = []  # maximum absolute error
    
    test_points_seq = np.arange(begin,end,interval).tolist() # Points you start your test predictions from
    
    for ep in range(epochs): 
        mse_test_loss_seq = []
        lincor_seq = []
        rmse_test_loss_seq = []
        mre_test_loss_seq = []
        mae_test_loss_seq = []
        maxae_test_loss_seq = []
        
        for w_i in test_points_seq:
            mse_test_loss = 0.0
            mre_test_loss = 0.0
            mae_test_loss = 0.0
            ae_test_loss = []
            rmse_test_loss = 0.0  #CHANGED
            our_predictions = []
            mid_data = []
            for step in range(begin//batch_size-2):   
                for pred_i in range(n_predict_once):
                
                    pred = predictions_over_time[ep][step][pred_i]
                
                    our_predictions.append(np.asscalar(pred))
        
                    mid_data.append(all_mid_data[w_i + pred_i])


                    mse_test_loss += 0.5*(pred-all_mid_data[w_i+pred_i])**2
                    mre_test_loss += abs((pred-all_mid_data[w_i+pred_i])/(all_mid_data[w_i+pred_i]))
                    mae_test_loss += abs(pred-all_mid_data[w_i+pred_i])
                    ae_test_loss.append(mae_test_loss)
       

        
        
        #lin_cor
#        mean_pred = our_predictions/n_predict_once
#        mean_mid_data = mid_data/n_predict_once
        
            cov = np.cov(our_predictions,mid_data)
            var_pred = np.var(our_predictions)
            var_mid_data = np.var(mid_data)
            lincor = cov/(math.sqrt(var_pred*var_mid_data))
        
 
            mse_test_loss /= n_predict_once
            mre_test_loss /= n_predict_once
            mae_test_loss /= n_predict_once
            maxae_test_loss = max(ae_test_loss)
            rmse_test_loss = math.sqrt(mse_test_loss)
        
            lincor_seq.append(lincor)
            mse_test_loss_seq.append(mse_test_loss)
            mre_test_loss_seq.append(mre_test_loss)
            mae_test_loss_seq.append(mae_test_loss)
            rmse_test_loss_seq.append(rmse_test_loss)
            maxae_test_loss_seq.append(maxae_test_loss)

              
        current_lincor = np.mean(lincor_seq)   
        current_test_mse = np.mean(mse_test_loss_seq)
        current_test_mre = np.mean(mre_test_loss_seq)
        current_test_mae = np.mean(mae_test_loss_seq)
        current_test_rmse = np.mean(rmse_test_loss_seq) #CHANGED
        current_test_maxae = np.mean(maxae_test_loss_seq)
      
      
        
        test_lincor_ot.append(current_lincor)  
        test_mse_ot.append(current_test_mse)
        test_mre_ot.append(current_test_mre)
        test_rmse_ot.append(current_test_rmse)
        test_mae_ot.append(current_test_mae)
        test_maxae_ot.append(current_test_maxae)
      
    return test_lincor_ot, test_mse_ot, test_mre_ot, test_rmse_ot, test_mae_ot, test_maxae_ot
                
        
            