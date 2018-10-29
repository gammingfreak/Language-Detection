#%% Step 4: Load Model and score sentences

import kenlm

model_ita = kenlm.LanguageModel('/Users/Cathy/Documents/code/ML-LD/ita.klm')
model_fra = kenlm.LanguageModel('/Users/Cathy/Documents/code/ML-LD/fra.klm')
model_por = kenlm.LanguageModel('/Users/Cathy/Documents/code/ML-LD/por.klm')
model_eng = kenlm.LanguageModel('/Users/Cathy/Documents/code/ML-LD/eng.klm')
model_spa = kenlm.LanguageModel('/Users/Cathy/Documents/code/ML-LD/spa.klm')
model_deu = kenlm.LanguageModel('/Users/Cathy/Documents/code/ML-LD/deu.klm')


score = np.full([n_test,n_lang],np.nan)
perplexity = np.full([n_test,n_lang],np.nan)

for i in range(n_test):
    score[i,0] = model_ita.score(data_test['sent'][i+1])
    perplexity[i,0] = model_ita.perplexity(data_test['sent'][i+1])
    score[i,1] = model_fra.score(data_test['sent'][i+1])
    perplexity[i,1] = model_fra.perplexity(data_test['sent'][i+1])
    score[i,2] = model_por.score(data_test['sent'][i+1])
    perplexity[i,2] = model_por.perplexity(data_test['sent'][i+1])
    score[i,3] = model_eng.score(data_test['sent'][i+1])
    perplexity[i,3] = model_eng.perplexity(data_test['sent'][i+1])
    score[i,4] = model_spa.score(data_test['sent'][i+1])
    perplexity[i,4] = model_spa.perplexity(data_test['sent'][i+1])
    score[i,5] = model_deu.score(data_test['sent'][i+1])
    perplexity[i,5] = model_deu.perplexity(data_test['sent'][i+1])


pred = np.argmax(score,axis=1)
pred_lang = list()

for i in range(n_test):
    if pred[i] == 0:
        pred_lang.append('ita')
    elif pred[i] == 1:
        pred_lang.append('fra')
    elif pred[i] == 2:
        pred_lang.append('por')
    elif pred[i] == 3:
        pred_lang.append('eng')
    elif pred[i] == 4:
        pred_lang.append('spa')
    else:
        pred_lang.append('deu')