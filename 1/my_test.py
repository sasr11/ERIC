from tqdm import tqdm
import sys
import time
import random
def func_1():
    # 测试进度条
    
    pass
        

def main(args, config, logger: Logger, run_id: int, dataset: DatasetLocal):
    T                                = Trainer(config=config, args= args, logger= logger)

    model, optimizer, loss_func      = T.init(dataset)   # model of current split
    custom                           = config.get('custom', False)  # False
    pbar                             = tqdm(range(config['epochs']), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # 进度条
    validation_data                  = None
    patience_cnt 		             = 0
    maj_metric 			             = "micro"   # or macro
    best_mse_metric                  = 100000.0
    best_metric 	  	             = 0
    best_metric_epoch 	             = -1 # best number on dev set
    report_mse_test 	             = 0
    report_rho_test                  = 0
    report_prec_at_10_test           = 0
    best_val_mse                     =  100000.
    best_val_tau                     = -100000.
    best_val_rho                     = -100000.
    best_val_p10                     = -100000.
    best_val_p20                     = -100000.
    best_val_epoch                   = -1
    loss_list                        = []
    monitor                          = config['monitor']  # mse
    best_val_paths                   = [None        , None        , None        , None        , None        ]
    best_val_metric                  = [best_val_mse, best_val_rho, best_val_tau, best_val_p10, best_val_p20]
    b_epoch                          = 0
    if config['save_best']:  # True  结果保存路径
        PATH_MODEL                   = os.path.join(os.path.join(os.getcwd(),'model_saved'), args.dataset, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))    
    for epoch in pbar:  # x/30000  x/50
        if not custom:  # True
            batches                  = dataset.create_batches_all(config)   # 所有图对
        else:
            batch_feature_1, batch_adj_1, batch_mask_1, batch_feature_2, batch_adj_2, batch_mask_2, batch_ged = dataset.custom_dataset.get_training_batch()
        main_index                   = 0
        loss_sum                     = 0
        total_loss_sum               = 0 
        # 训练
        for batch_pair in batches:  # 遍历所有图对，每次128个图对
            data                     = dataset.transform_batch(batch_pair, config)
            target                   = data["target"].cuda()
            model, loss              = T.train(data, model, loss_func, optimizer, target)   # 图对数据、模型、
            main_index               = main_index + batch_pair[0].num_graphs               
            loss_sum                 = loss_sum + loss                                    
        loss                         = loss_sum / main_index                              
        loss_list.append(loss)

        # 验证集测试
        if config['use_val']:  # True
            if epoch >= config['iter_val_start'] and epoch % config['iter_val_every'] ==0:  # 29000  10
                model.eval()
                val_mse, val_rho, val_tau, val_prec_at_10, val_prec_at_20 = T.evaluation(dataset.val_graphs, dataset.training_graphs, model, loss_func, dataset, validation=True)
                logger.log("Validation Epoch = {}, MSE = {}(e-3), rho = {}, tau={}, prec_10 = {}, prec_20 = {}".format(epoch, val_mse*1000, val_rho, val_tau, val_prec_at_10, val_prec_at_20))
                if not config.get('save_best_all', False):  # False
                    if best_mse_metric                >= val_mse:
                        best_mse_metric               = val_mse
                        best_val_epoch                = epoch
                        best_val_mse                  = val_mse
                        best_val_tau                  = val_tau
                        best_val_rho                  = val_rho
                        best_val_p10                  = val_prec_at_10
                        best_val_p20                  = val_prec_at_20
                        if config['save_best']:
                            best_val_model_path       = save_best_val_model(config, args.dataset, model, PATH_MODEL)
                else:
                    current_metric                    = [val_mse        , val_rho     , val_tau     , val_prec_at_10,  val_prec_at_20, epoch]
                    best_val_metric, best_val_paths, b_epoch = save_best_val_model_all(config, args.dataset, model, PATH_MODEL, current_metric, best_val_metric, best_val_paths, b_epoch, validation=True)
                    best_mse_metric                   = best_val_metric[0]
                    best_val_mse                      = best_val_metric[0]
                    best_val_rho                      = best_val_metric[1]
                    best_val_tau                      = best_val_metric[2]
                    best_val_p10                      = best_val_metric[3]
                    best_val_p20                      = best_val_metric[4]
                    best_val_epoch                    = b_epoch

        # 输出1-29999轮的结果
        if epoch != config['epochs']-1:  # 如果epoch没到29999
            postfix_str = "<Epoch %d> [Train Loss] %.5f"% ( 
                            epoch ,      loss)
            # pbar.set_postfix_str(postfix_str)
        # 输出30000轮的结果  
        elif epoch == config['epochs'] and config.get('show_last', False): 
            mse, rho, tau, prec_at_10, prec_at_20 = T.evaluation(dataset.testing_graphs, dataset.training_graphs, model, loss_func, dataset)
            best_mse_metric                       = mse
            best_metric_epoch                     = epoch
            report_mse_test                       = mse
            report_rho_test                       = rho
            report_tau_test                       = tau
            report_prec_at_10_test                = prec_at_10
            report_prec_at_20_test                = prec_at_20
            
            postfix_str = "<Epoch %d> [Train Loss] %.4f [Cur Tes %s] %.4f <Best Epoch %d> [Best Tes mse] %.4f [rho] %.4f [tau] %.4f [prec_at_10] %.4f [prec_at_20] %.4f " % ( 
                            epoch ,      loss,         monitor,      eval(monitor),  
                            best_metric_epoch ,report_mse_test, report_rho_test,report_tau_test,report_prec_at_10_test,report_prec_at_20_test)
        else:
            postfix_str = "<Epoch %d> [Train Loss] %.5f"% ( 
                epoch ,      loss)
            
        if not args.train_first:  # False
            mse, rho, tau, prec_at_10, prec_at_20 = T.evaluation(dataset.testing_graphs, dataset.training_graphs, model, loss_func, dataset)  # return 2 list, 
            if monitor                         == 'mse':   # *↓
                if mse                         <= best_mse_metric:
                    best_mse_metric             = mse
                    best_metric_epoch           = epoch
                    report_mse_test             = mse
                    report_rho_test             = rho
                    report_tau_test             = tau
                    report_prec_at_10_test      = prec_at_10
                    report_prec_at_20_test      = prec_at_20
                    patience_cnt                = 0  
                else:
                    patience_cnt               += 1
            elif monitor in ['rho', 'tau', 'prec_at_10', 'prec_at_20']:   # *↑
                current_metric                  = eval(monitor)
                if best_metric                 <= current_metric: 
                    best_metric                 = current_metric
                    best_metric_epoch           = epoch
                    report_mse_test             = mse
                    report_rho_test             = rho
                    report_tau_test             = tau
                    report_prec_at_10_test      = prec_at_10
                    report_prec_at_20_test      = prec_at_20
                    patience_cnt                = 0  
                else:
                    patience_cnt               += 1               

            if config['patience'] > 0 and patience_cnt >= config['patience']:
                break

            postfix_str = "<Epoch %d> [Train Loss] %.4f [Cur Tes %s] %.4f <Last Epoch %d> [Last Tes mse] %.4f [rho] %.4f [tau] %.4f [prec_at_10] %.4f [prec_at_20] %.4f " % ( 
                            epoch ,      loss,         monitor, eval(monitor),  
                            best_metric_epoch ,report_mse_test, report_rho_test,report_tau_test,report_prec_at_10_test,report_prec_at_20_test)

        pbar.set_postfix_str(postfix_str)
    
    # 测试集进行评估
    logger.add_line()
    logger.log("start testing using best val model")
    if not config.get('save_best_all', False):  # False
        model.load_state_dict(torch.load(best_val_model_path))
        test_mse, test_rho, test_tau, test_prec_at_10, test_prec_at_20 = T.evaluation(dataset.testing_graphs, dataset.trainval_graphs, model, loss_func, dataset)
    else:
        met_test                                                       = load_model_all(dataset, model, loss_func, best_val_paths, T)
        test_mse, test_rho, test_tau, test_prec_at_10, test_prec_at_20 = met_test
    best_val_result = {
        'best_val_epoch': best_val_epoch,
        'best_val_mse'  : best_val_mse,
        'best_val_tau'  : best_val_tau,
        'best_val_rho'  : best_val_rho,
        'best_val_p10'  : best_val_p10,
        'best_val_p20'  : best_val_p20
    }
    return model, best_val_epoch , test_mse, test_rho, test_tau, test_prec_at_10, test_prec_at_20, loss, PATH_MODEL, best_val_result



if __name__ == "__main__":
    func_1()
    pass