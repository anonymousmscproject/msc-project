import numpy as np
import pandas as pd
import torch
from collections import namedtuple

### SCRIPTS TO LOAD DATASETS: ADULT, COMPAS, ARHYTHMIA, DRUG, GERMAN ###
### TOY DATASET FOR META DISTRIBUTION ####
#############################################################

def adult(file_path1="./data/adult/adult.data", file_path2="./data/adult/adult.test", very_small=False, small=False):
    '''
    Features:

    0. Age
    1. Employment type: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    2. Final weight
    3. Education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th,
        Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    4. Years in Education,
    5. Marital status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
        Married-spouse-absent, Married-AF-spouse.
    6. Occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
        Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
        Protective-serv, Armed-Forces.
    7. Relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    8. Race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    9. Sex: Female, Male.
    10. Capital Gain
    11. Capital Loss
    12. Working Hours
    13. Native Country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc),
        India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
        Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,
        Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    
    14. (LABEL) Salary: <=50K, >50K
    
    '''
    f = open(file_path1)
    data_train = pd.read_csv(f, header=None, na_values=' ?')
    len_train = data_train[0].shape[0]

    f2 = open(file_path2)
    data_test = pd.read_csv(f2, header=None, na_values=' ?')
    
    data = pd.concat([data_train, data_test])
    data = data.dropna()
    
    # replace categories with a number
    cat_cols = [1, 3, 5, 6, 7, 8, 9, 13, 14] #cols with categorical data
    names = [[]]
    for i in cat_cols:
        cat, num = np.unique(data[i], return_inverse=True)
        names.append(cat)
        data[i] = num
        
    # dictionary for category columns which are now assigned a number    
    names.pop(0)
    category_dict = []
    for i in range(len(names)):
        for j in range(len(names[i])):
            category_dict.append({j:names[i][j]})
    
    data_vals = data.values
    labels_col = np.array(data_vals)[:, -1]

    labels = []
    for i in labels_col:
        if i == 0:
            labels.append(-1.0)
        else:
            labels.append(1.0)
            
    labels = np.array(labels)
    data_vals = data_vals[:, :-1]

    if very_small:
        data_tr = namedtuple('_', 'data, labels')(data_vals[:len_train // 250], labels[:len_train // 250])
        data_te = namedtuple('_', 'data, labels')(data_vals[len_train:len_train+50], labels[len_train:len_train+50])
        print("Ver small subset of Adult dataset loaded")

    elif small:
        data_tr = namedtuple('_', 'data, labels')(data_vals[:len_train // 2], labels[:len_train // 2])
        data_te = namedtuple('_', 'data, labels')(data_vals[len_train:], labels[len_train:])
        print("Subset of Adult dataset loaded")

    else:
        data_tr = namedtuple('_', 'data, labels')(data_vals[:len_train], labels[:len_train])
        data_te = namedtuple('_', 'data, labels')(data_vals[len_train:], labels[len_train:])
        print("Full Adult dataset loaded")

    return data_tr, data_te

#############################################################


def compas(file_path="./data/compas/compas-scores-two-years-violent.csv", compas_recid=True):

    '''Can either use the compas recidivism decision as the label or the two year recidivism as the label. Set compas_recid=True to use the former
    0: sex, 1: age, 2: age_cat 3: race'''

    f = open(file_path)
    data = pd.read_csv(f)

    data.replace(['Other', 'African-American','Caucasian', 'Hispanic', 'Asian', 'Native American'],
                 ['Non-White','Non-White', 'White','Non-White','Non-White','Non-White'], inplace=True)
    
    data = data.drop(['id'], axis=1)

    cat_cols = ['sex', 'age_cat', 'race', 'c_charge_degree', 'v_score_text'] #cols with categorical data
    names = [[]]
    for i in cat_cols:
        cat, num = np.unique(data[i], return_inverse=True)
        names.append(cat)
        data[i] = num
        
    # dictionary for category columns which are now assigned a number    
    names.pop(0)
    category_dict = []
    for i in range(len(names)):
        for j in range(len(names[i])):
            category_dict.append({j:names[i][j]})

    data.replace(['Low', 'Medium','High'],
                 [1,2,0], inplace=True)

    data = data.drop(['days_b_screening_arrest', 'r_days_from_arrest'], axis=1)

    data = data.dropna()

    if compas_recid:
        
        data = data.drop(['two_year_recid'], axis=1)  
        labels_col = data['is_violent_recid'].values
        labels = []
        for i in labels_col:
            if i == 1:
                labels.append(-1.0) # did recid
            else:
                labels.append(1.0) # did not

        labels = np.array(labels)
        
        data = data.drop(['is_violent_recid'], axis=1)
        data_vals = data.values

        data = namedtuple('_', 'data, labels')(data_vals, labels)
        
    else:
        
        data = data.drop(['is_violent_recid'], axis=1)
        data_vals = data.values

        labels_col = np.array(data_vals)[:, -1]
        labels = []
        for i in labels_col:
            if i == 1:
                labels.append(-1.0) # did recid
            else:
                labels.append(1.0) # did not 

        labels = np.array(labels)

        data_vals = data_vals[:, :-1]

        data = namedtuple('_', 'data, labels')(data_vals, labels) 

    return data

#############################################################

def arrhythmia(file_path="./data/arrhythmia/arrhythmia.data"):
    #gender attribute 2 (col 1)
    f = open(file_path)
    data = pd.read_csv(f, header=None, na_values='?')
    len_data = data[0].shape[0]
    data = data.dropna()

    data_vals = data.values
    labels_col = np.array(data_vals)[:, -1]

    labels = []
    for i in labels_col:
        if i == 1:
            labels.append(1.0) # normal
        else:
            labels.append(-1.0) # if arryhtmia

    labels = np.array(labels)
    data_vals = data_vals[:, :-1]   
    data = namedtuple('_', 'data, labels')(data_vals, labels)

    return data


#############################################################



def drug(file_path="./data/drug/drug_consumption.data", drug_col=25):
    # col 5 us ethncity 
    # drug col 25 =  heroin
    f = open(file_path)
    data = pd.read_csv(f, header=None)
    data = data.dropna()
    data[5].replace([0.126, -0.31685,0.1144, -0.22166,-0.50212,-1.10702],
                 [1, 0, 1, 1, 1, 1], inplace=True)
    data_vals = data.values
    data_vals = data_vals[:, :13]

    labels_col = data[drug_col].values #default heroin
    labels = []
    for i in labels_col:
        if i == 'CL0':
            labels.append(1.0) # never taken
        else:
            labels.append(-1.0) # used

    labels = np.array(labels)

    data = namedtuple('_', 'data, labels')(data_vals, labels)

    return data


#############################################################





def german(file_path="./data/german/german.data"):
    f = open(file_path)
    data = pd.read_csv(f, sep=" ",header=None, engine='python')
    data = data.dropna()

    data.replace(['A91', 'A92','A93', 'A94'],
                 [1, 0, 1, 1], inplace=True) 
                 
    # gender, col 8

    # replace categories with a number
    cat_cols = [0, 2, 3, 5, 6, 9, 11, 13, 14, 16, 18, 19] #cols with categorical data
    names = [[]]
    for i in cat_cols:
        cat, num = np.unique(data[i], return_inverse=True)
        names.append(cat)
        data[i] = num
        
    # dictionary for category columns which are now assigned a number    
    names.pop(0)
    category_dict = []
    for i in range(len(names)):
        for j in range(len(names[i])):
            category_dict.append({j:names[i][j]})

    data_vals = data.values

    labels_col = np.array(data_vals)[:, -1]
    labels = []
    for i in labels_col:
        if i == 1:
            labels.append(1.0) # good
        else:
            labels.append(-1.0) # bad

    labels = np.array(labels)

    data_vals = data_vals[:, :-1]

    data = namedtuple('_', 'data, labels')(data_vals, labels)

    return data


##################

def toy_data(total_number=50, mean1=None, mean2=None, theta=None, 
             std1=0.1, std2=None, std3=None, std4=None,
             feature_imbalance=0.6, class_imbalance=0.8,
             dim=2, max_deg__group_b=30, 
             dist_center_a=1., dist_center_b=1.):
    
    if feature_imbalance is None:
        a = total_number * 0.5
        b = total_number * 0.5
        
    else:
        a = total_number * feature_imbalance
        b = total_number * (1 - feature_imbalance)
    
    if class_imbalance is None: 
        a_p = int(a * 0.5)
        a_n = int(a * 0.5)
        b_p = int(b * 0.5)
        b_n = int(b * 0.5)
    
    else:
        a_p = int(a * class_imbalance)
        a_n = int(a * (1 - class_imbalance))
        b_p = int(b * (1 - class_imbalance))
        b_n = int(b * class_imbalance)
  
    N_A_positive=(a_p,)
    N_B_positive=(b_p,)
    N_A_negative=(a_n,)
    N_B_negative=(b_n,)
    
    # group A
    if mean1 is None: mean1 = torch.cat((torch.ones(1), torch.zeros(dim - 1)), dim=0)
    if mean2 is None: mean2 = torch.cat((-torch.ones(1), torch.zeros(dim - 1)), dim=0)
        
    # group B
    if theta is None: theta = (torch.rand(1) * max_deg__group_b * 3.1416) / 180
    mean_b1 = torch.cat((theta.cos(), theta.sin(), torch.zeros(dim - 2)))
    mean_b2 = - mean_b1.clone().detach()

    if std2 is None: std2 = std1
    if std3 is None: std3 = std1
    if std4 is None: std4 = std1

    if N_A_negative is None: N_A_negative = N_A_positive
    if N_B_negative is None: N_B_negative = N_B_positive

    xs, ys, ats = [], [], []

    for nap, nan, nbp, nbn in zip(N_A_positive, N_A_negative, N_B_positive, N_B_negative):
        
        # attribute
        ats.append(torch.cat((torch.zeros(nap + nan, 1), torch.ones(nbp + nbn, 1)), dim=0))
        
        # data
        xs.append(
            torch.cat((dist_center_a * mean1 + std1 * torch.randn((nap, dim)),
                       dist_center_a * mean2 + std2 * torch.randn((nan, dim)),
                       dist_center_b * mean_b1 + std3 * torch.randn((nbp, dim)),
                       dist_center_b * mean_b2 + std4 * torch.randn((nbn, dim)),), 
                       dim=0))
        
        # labels
        ys.append(torch.cat((torch.ones(nap, 1),-torch.ones(nan, 1),
                             torch.ones(nbp, 1),-torch.ones(nbn, 1),), 
                             dim=0))
        
    xs = [torch.cat((x, at), dim=-1) for x, at in zip(xs, ats)]   
        
    info = {'mean1': mean1,
            'mean2': mean2,
            'mean_b1': mean_b1,
            'mean_b2': mean_b2,
            'theta': theta,
            'std1': std1,
            'std2': std2,
            'std3': std3,
            'std4': std4,
            'class imbalance': class_imbalance,
            'feature imbalance': feature_imbalance}

    ts = []
    for t in ys:
        tmp = t.numpy()
        ts.append(tmp)
    ts = np.array(ts)

    labels = ts[0,:,0]

    xs_n = []
    for i in xs:
        tmp = i.numpy()
        xs_n.append(tmp)
    xs_n = np.array(xs_n)
    train_data = xs_n[0,:,:]
    train = [train_data,labels]
        
    return mean1, mean2, std1, std2, train, train_data, labels