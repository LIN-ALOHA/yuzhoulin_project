# yuzhoulin_project
this repo is about demo and evaluation on interpreting image-based malware dataset on machine-learning by varities of explaining methods.
## background
the project is aimed at proposing an interpreble image-based malware detection architecture called IEMD, namely image-based ensemble malware detection.
Currently, deep-learning-based methods towards malware detection lack interpretability, which is far from practising them in the real industry area.
Our work aims at resolving and alleviating this bad trend and simultaneously creating some useful metrics for evaluating malware detection interpretability quantitatively.
## usage
By operating a demo on our work, you need to firstly employ comprehensive.py file for the further use.
```
  python comprehensive.py -[args]
```
### methods
We have major methods for explaining deep-learning malware detection, namely, deep-lift, guided-g
rad CAM, lemna, and smooth-grad.
### datasets
We use Malimg, IoT malware, for more information, please contact:
https://pan.baidu.com/s/1qXfiPg_QE_t46dlsG5tbWg
passwd: `37eq`
Inside our entire work, in baidu cloud dst, we have everything required for you to conduct an entire demo for deeply understand our work.
### IEMD
#### our framework
![image](https://github.com/LIN-ALOHA/yuzhoulin_project/blob/lester/IEMD_framework.png)

#### contribution
* ensemble-learning architecture
* deep Taylor decomposition embedded on ensemble structure with/without reweighting technique
* iDrop normalization technique here adjusted the parameter with 0.6, according to the tuning process depicted in ariticle
Lin, Y., & Chang, X. (2021). Towards Interpretable Ensemble Learning for Image-based Malware Detection. arXiv preprint arXiv:2101.04889.
#### experimental results partial presentation
* detailed tuning curves on idrop
* 
![image](https://github.com/LIN-ALOHA/yuzhoulin_project/blob/lester/tunung_idrop.png)

* detailed comparative experiments on different malware detection interpretability work by using pixel-reversing tricks(in window-size, namely focal reversing)
* 
![image](https://github.com/LIN-ALOHA/yuzhoulin_project/blob/lester/comparative_explain.png)

#### our metrics and relative evaluation experiment results
methods|accuracy|precision rate|recall rate|f1-score
------|-------|-------|----|----
IMCFN(DL)|0.98|0.98|0.98|0.98
dl|0.97|n/a|0.88|n/a
knn on GIST features|0.97|n/a|n/a|n/a
cnn+svm on entropy graphs|0.99|n/a|n/a|0.99
IEMD(our work)|0.99|0.99|0.99|0.99

### demo
* python script to conduct some demos for research
* patches below(note: this is from comprehensive_transfer_train.py which `is not` in the github `but in the baidu dist`,)
```python
if __name__=='__main__':
    # data_dir = r'C:\Users\lin\Desktop\gradduate_design\malware_grayscale_interpret\IoT_malware_grayscale_images'
    data_dir=r'C:\Users\lin\Desktop\gradduate_design\malware_grayscale_interpret\malware_greyscale_families'
    # Models to choose from [resnet50, alexnet, vgg11_bn,vgg16, squeezenet, densenet121, inception v3]
    # model_choice = ['resnet50','inception_v3','densenet121','IIMD']
    # model_choice=['IIMD']
    # model_choice=['resnet18','resnet50','vgg11_bn','vgg16','squeezenet','densenet121',
    #               'inception v3','alexnet','IIMD']
    # model_choice=['resnext50','resnext101']
    model_choice=['RRV_complete']
    # Number of classes in the dataset
    num_classes = 25

    # Batch size for training (change depending on how much memory you have)
    batch_size = 2

    # Number of epochs to train for
    num_epochs = 20

    #k-fold
    k_fold = 10

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params

    button='train'# ['train', 'interpret'，‘test'，‘DTD_train]

    ###################################
    k_fold_accuracy=[]
    if button=='train':
        for index,model_name in enumerate(model_choice):
            if model_name=='resnext50' or model_name=='resnext101':
                feature_extract = False
            else:
                feature_extract = True
            # Initialize the model for this run
            print('transfer train {}'.format(model_name))
            model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True,input_size_used=224)

            # Print the model we just instantiated
            data_transforms = {
                'train': transforms.Compose([
                    transforms.Resize(input_size),
                    # transforms.RandomResizedCrop(input_size),#data preprocessing need proved
                    # transforms.RandomHorizontalFlip(),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'test': transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }

            print("Initializing Datasets and Dataloaders...")

            # Create training and validation datasets
            image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
            # Create training and validation dataloaders
            dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in ['train', 'test']}

            # Detect if we have a GPU available
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(device)
            model_ft = model_ft.to(device)

            # Gather the parameters to be optimized/updated in this run. If we are
            #  finetuning we will be updating all parameters. However, if we are
            #  doing feature extract method, we will only update the parameters
            #  that we have just initialized, i.e. the parameters with requires_grad
            #  is True.
            if model_name=='RRV_complete':
                class_names = image_datasets['train'].classes
                path=r'C:\Users\lin\Desktop\gradduate_design\model\malimg_model_tl\RRV_complete.pt'
                # model_ft,best_val_history=IIMD_train(model_ft,path,dataloaders_dict,device,10,batch_size,model_name)
                model_ft=torch.load(path)
                # k_fold_accuracy.append(best_val_history)
                # visualize_outcome_inOneModel(best_val_history,model_name)
                RRV_test(model_ft, dataloaders_dict, device, class_names,model_name)  # 如果要测试，需要把dataloader改为有序取出
                # plot_conf_matrix(dataloaders_dict,model_ft,class_names,device,model_name)
            elif model_name=='IIMD':
                path=r'C:\Users\lin\Desktop\gradduate_design\model\IOT_model_tl\IIMD_IOT.pt'
                # path=r'C:\Users\lin\Desktop\gradduate_design\model\malimg_model_tl\IIMD.pt'
                # iimd,best_val_history=IIMD_train(model_ft,path,dataloaders_dict,device,k_fold,batch_size,model_name)
                # iimd=torch.load(path)
                iimd=torch.load(path)
                class_names = image_datasets['test'].classes
                transfer_test(iimd,dataloaders_dict,device,class_names)#如果要测试，需要把dataloader改为有序取出
                k_fold_accuracy.append(best_val_history)
                visualize_outcome_inOneModel(best_val_history, model_name)  # 绘制一个模型的k-fold曲线
            else:
                params_to_update = model_ft.parameters()
                print("Params to learn:")
                if feature_extract:
                    params_to_update = []
                    for name,param in model_ft.named_parameters():
                        if param.requires_grad == True:
                            params_to_update.append(param)
                            print("\t",name)
                else:
                    for name,param in model_ft.named_parameters():
                        if param.requires_grad == True:
                            print("\t",name)

                # Observe that all parameters are being optimized
                optimizer_ft = optim.SGD(params_to_update, lr=0.01, momentum=0.9)
                # optimizer_ft=optim.Adam(params_to_update,lr=0.01,betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)
                # Setup the loss fxn
                criterion = nn.CrossEntropyLoss()
                #modify the lr step
                # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
                # exp_lr_scheduler=lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=3,
         # verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
                exp_lr_scheduler=lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=10, eta_min=0, last_epoch=-1)
                # Train and evaluate
                model_ft, best_val_history = k_fold_corss_validation_train_model(batch_size,exp_lr_scheduler,model_name,k_fold,model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
                # model=train_model(exp_lr_scheduler,model_name,model_ft,dataloaders_dict,criterion,optimizer_ft,40,is_inception=(model_name=="inception"))
                torch.save(model_ft,r'C:\Users\lin\Desktop\gradduate_design\model\malimg_model_tl\{}_tl_malimg.pt'.format(model_name))
                k_fold_accuracy.append(best_val_history)
                visualize_outcome_inOneModel(best_val_history,model_name)#绘制一个模型的k-fold曲线
                #Detection
                # model_ft=torch.load(r'C:\Users\lin\Desktop\gradduate_design\model\resnet50_mal_iot_2.pt')
                class_names = image_datasets['test'].classes
                plot_conf_matrix(dataloaders_dict, model_ft, class_names, device,model_name)#绘制一个模型的侦测混淆矩阵和曲线
                del input_size,data_transforms,image_datasets,dataloaders_dict,device,model_ft,params_to_update,optimizer_ft,criterion
                gc.collect()
        visualize_k_fold_outcome(k_fold_accuracy, model_choice)  # 绘制所有模型的k-fold曲线
    elif button=='plot':
        input_size=224
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(input_size),
                # transforms.RandomResizedCrop(input_size),#data preprocessing need proved
                # transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        print("Initializing Datasets and Dataloaders...")

        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                          ['train', 'test']}
        # Create training and validation dataloaders
        dataloaders_dict = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) for x
            in ['train', 'test']}
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        class_names = image_datasets['test'].classes
        plot_detect_malimg(model_choice,device,dataloaders_dict,class_names)
    elif button=='interpret':
        print('interpret')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_size = 224
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(input_size),
                # transforms.RandomResizedCrop(input_size),#data preprocessing need proved
                # transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # print("Initializing Datasets and Dataloaders...")
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                          ['train', 'test']}
        # Create training and validation dataloaders
        dataloaders_dict = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) for x
            in ['train', 'test']}
        class_names = image_datasets['test'].classes
        path = r'C:\Users\lin\Desktop\gradduate_design\model\malimg_model_tl\RRV_complete.pt'
        path_2=r'C:\Users\lin\Desktop\gradduate_design\model\malimg_model_tl\resnet50_tl_malimg.pt'
        model_ft = torch.load(path)
        print(model_ft)
        # model_ft=resnet50_convert_format(model_ft)
        model_ft=model_ft.to(device)
        path_right=r'C:\Users\lin\Desktop\gradduate_design\Interpret_malware_detection\results\DTD_results\malimg_DTD\RRV_right'
        path_wrong=r'C:\Users\lin\Desktop\gradduate_design\Interpret_malware_detection\results\DTD_results\malimg_DTD\RRV_wrong'
        # DTD_interpret(model_ft, 5000, 'resnet', dataloaders_dict, device, class_names, path_right, path_wrong)
        DTD_ensemble_interpret(model_ft,5000,dataloaders_dict,device,class_names,path_right,path_wrong,batch_size)
    elif button=='DTD_train':
        print('begin to use DTD to enhance training robustness...')
        model=torch.load(r'C:\Users\lin\Desktop\gradduate_design\model\malimg_model_tl\RRV_complete.pt')
        path=r'C:\Users\lin\Desktop\gradduate_design\model\malimg_model_tl\RRV_complete_DTD_finetune_firstlayer.pt'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('fine-tune RRV')
        model=RRV_fine_tune_first_layer(model)
        model = model.to(device)
        print('success')
        input_size = 224
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                          ['train', 'test']}
        dataloaders_dict = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) for x
            in ['train', 'test']}
        class_names = image_datasets['test'].classes
        # DTD_train_data,DTD_train_label,DTD_test_data,DTD_test_label=DTD_prepare_dropout(dataloaders_dict,model,5000,class_names,device)
        # torch.save(DTD_train_data , r'C:\Users\lin\Desktop\gradduate_design\model\dataloader\RRV\DTD\train_data.pt')
        # torch.save(DTD_train_label , r'C:\Users\lin\Desktop\gradduate_design\model\dataloader\RRV\DTD\train_label.pt')
        # torch.save(DTD_test_data , r'C:\Users\lin\Desktop\gradduate_design\model\dataloader\RRV\DTD\test_data.pt')
        # torch.save(DTD_test_label , r'C:\Users\lin\Desktop\gradduate_design\model\dataloader\RRV\DTD\test_label.pt')
        DTD_train_data=torch.load(r'C:\Users\lin\Desktop\gradduate_design\model\dataloader\RRV\DTD\train_data.pt')
        DTD_train_label=torch.load(r'C:\Users\lin\Desktop\gradduate_design\model\dataloader\RRV\DTD\train_label.pt')
        DTD_test_data=torch.load(r'C:\Users\lin\Desktop\gradduate_design\model\dataloader\RRV\DTD\test_data.pt')
        DTD_test_label=torch.load(r'C:\Users\lin\Desktop\gradduate_design\model\dataloader\RRV\DTD\test_label.pt')
        print('saved')
        dataloader_dict, labelloader_dict = DTD_dataloader(DTD_train_data, DTD_train_label, DTD_test_data,
                                                           DTD_test_label,batch_size)
        #根据DTD的可解释映射定性抑制输入神经元，训练RRV基分类器输入卷积层和二级全连接层，10-折交叉验证学习
        model,best_acc=RRV_DTD_train(model,path,dataloader_dict,labelloader_dict,device,10,batch_size,model_choice[0],class_names)
        RRV_test(model,dataloaders_dict,labelloader_dict,device,class_names,model_choice[0])#测试训练好的RRV_DTD的效果
        visualize_outcome_inOneModel(best_acc, model_choice[0])#显示k-折交叉验证训练过程曲线
    elif button=='test':
        model = torch.load(r'C:\Users\lin\Desktop\gradduate_design\model\malimg_model_tl\RRV_complete.pt')
        RRV_fine_tune_first_layer(model)
        print(model)
```
* 
