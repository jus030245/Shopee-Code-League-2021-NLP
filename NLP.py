#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time
import re
import torch


# In[2]:


train = pd.read_csv('/Users/liu/Downloads/Address Elements Extraction Dataset/train.csv')
test = pd.read_csv('/Users/liu/Downloads/Address Elements Extraction Dataset/test.csv')


# In[3]:


train[['POI', 'street']] = train['POI/street'].str.split('/', 1, expand=True)
#train['POI/street'].replace('/', np.nan, inplace=True)
#train['POI'].replace('', np.nan, inplace=True)
#train['street'].replace('', np.nan, inplace=True)
train.head()


# In[8]:


def str_comp(row):
    if pd.isnull(row['POI']):
        return np.nan
    else:
        return (row['POI'] in row['raw_address'])
train['POI_comp'] = train.apply(str_comp, axis=1)


# In[9]:


def str_comp(row):
    if pd.isnull(row['street']):
        return np.nan
    else:
        return (row['street'] in row['raw_address'])
train['street_comp'] = train.apply(str_comp, axis=1)


# In[84]:


# from tensorflow.keras.preprocessing.text import Tokenizer
# spliter = Tokenizer(num_words=None, filters='!"#$%()*+,-/:;<=>?@[\]^_`{|}~ ', lower=True, split=' ', char_level=False, 
# oov_token=None, document_count=0)


# In[6]:


def transform(row):
    
    df = pd.DataFrame({'id':row['id'], 'word':re.split(r"[\[\]()\"*\{\} :,]",row['raw_address']), 'tag':'O'})

    try:
        POI = re.split(r"[\[\]()\"*\{\} :,]",row['POI'])
        df.loc[df[df['word'] == POI[0]].index,'tag'] = 'B-P'
        df.loc[df[df['word'].isin(POI[1:])].index,'tag'] = 'I-P'
    except:
        pass

    try:
        street = re.split(r"[\[\]()\"*\{\} :,]",row['street'])
        df.loc[df[df['word'] == street[0]].index,'tag'] = 'B-s'
        df.loc[df[df['word'].isin(street[1:])].index,'tag'] = 'I-s'
    except:
        pass

    return df


# In[10]:


train_ez = train[~((train['POI_comp'] == False)|(train['street_comp'] == False))]
train_hard = train[(train['POI_comp'] == False)|(train['street_comp'] == False)]


# In[14]:


np.save('ez_index',np.array(train_ez.index))
np.save('hard_index',np.array(train_hard.index))


# In[8]:


start = time.time()

train_ez_transformed = pd.DataFrame()

for i in range(train_ez.shape[0]):
    df = transform(train_ez.iloc[i])
    train_ez_transformed = pd.concat([train_ez_transformed,df])
train_ez_transformed = train_ez_transformed.reset_index(drop=True)

print('Time: ',time.time()-start)
train_ez_transformed.to_csv('train_ez_transformed_newjupyter.csv')


# In[2]:


train_ez_transformed = pd.read_csv('train_ez_transformed.csv')


# In[10]:


train_ez_transformed['word'] = train_ez_transformed['word'].astype('str')


# In[22]:


print('空.:',np.sum(train['POI/street'].apply(lambda x: ' .' in x)))
print('空.空:',np.sum(train['POI/street'].apply(lambda x: ' . ' in x)))
print('.空:',np.sum(train['POI/street'].apply(lambda x: '. ' in x)))


# In[18]:


train['POI/street'] = train['POI/street'].astype('str')


# In[19]:


train['check'] = train['POI/street'].apply(lambda x: ' .' in x)


# In[10]:


train_ez_transformed['tag'].value_counts()


# In[69]:


train = pd.read_csv('/Users/liu/Downloads/Address Elements Extraction Dataset/train.csv')
print('?:',np.sum(train['POI/street'].apply(lambda x: '?' in x)))
print('(:',np.sum(train['POI/street'].apply(lambda x: '(' in x)))
print('):',np.sum(train['POI/street'].apply(lambda x: ')' in x)))
print('_:',np.sum(train['POI/street'].apply(lambda x: '_' in x)))
print('.:',np.sum(train['POI/street'].apply(lambda x: '.' in x)))
print('%:',np.sum(train['POI/street'].apply(lambda x: '%' in x)))
print('>:',np.sum(train['POI/street'].apply(lambda x: '>' in x)))
print('<:',np.sum(train['POI/street'].apply(lambda x: '<' in x)))
print(']:',np.sum(train['POI/street'].apply(lambda x: ']' in x)))
print('[:',np.sum(train['POI/street'].apply(lambda x: '[' in x)))
print('|:',np.sum(train['POI/street'].apply(lambda x: '|' in x)))
print('{:',np.sum(train['POI/street'].apply(lambda x: '{' in x)))
print('}:',np.sum(train['POI/street'].apply(lambda x: '}' in x)))
print('&:',np.sum(train['POI/street'].apply(lambda x: '&' in x)))
print('$:',np.sum(train['POI/street'].apply(lambda x: '$' in x)))
print('@:',np.sum(train['POI/street'].apply(lambda x: '@' in x)))
print('#:',np.sum(train['POI/street'].apply(lambda x: '#' in x)))
print('*:',np.sum(train['POI/street'].apply(lambda x: '*' in x)))
print('~:',np.sum(train['POI/street'].apply(lambda x: '~' in x)))
print('^:',np.sum(train['POI/street'].apply(lambda x: '^' in x)))
print('":',np.sum(train['POI/street'].apply(lambda x: '"'in x)))
print(',:',np.sum(train['POI/street'].apply(lambda x: ','in x)))
print('::',np.sum(train['POI/street'].apply(lambda x: ':'in x)))


# In[68]:


train = pd.read_csv('/Users/liu/Downloads/Address Elements Extraction Dataset/train.csv')
print('?:',np.sum(train['raw_address'].apply(lambda x: '?' in x)))
print('(:',np.sum(train['raw_address'].apply(lambda x: '(' in x)))
print('):',np.sum(train['raw_address'].apply(lambda x: ')' in x)))
print('_:',np.sum(train['raw_address'].apply(lambda x: '_' in x)))
print('.:',np.sum(train['raw_address'].apply(lambda x: '.' in x)))
print('%:',np.sum(train['raw_address'].apply(lambda x: '%' in x)))
print('>:',np.sum(train['raw_address'].apply(lambda x: '>' in x)))
print('<:',np.sum(train['raw_address'].apply(lambda x: '<' in x)))
print(']:',np.sum(train['raw_address'].apply(lambda x: ']' in x)))
print('[:',np.sum(train['raw_address'].apply(lambda x: '[' in x)))
print('|:',np.sum(train['raw_address'].apply(lambda x: '|' in x)))
print('{:',np.sum(train['raw_address'].apply(lambda x: '{' in x)))
print('}:',np.sum(train['raw_address'].apply(lambda x: '}' in x)))
print('&:',np.sum(train['raw_address'].apply(lambda x: '&' in x)))
print('$:',np.sum(train['raw_address'].apply(lambda x: '$' in x)))
print('@:',np.sum(train['raw_address'].apply(lambda x: '@' in x)))
print('#:',np.sum(train['raw_address'].apply(lambda x: '#' in x)))
print('*:',np.sum(train['raw_address'].apply(lambda x: '*' in x)))
print('~:',np.sum(train['raw_address'].apply(lambda x: '~' in x)))
print('^:',np.sum(train['raw_address'].apply(lambda x: '^' in x)))
print('":',np.sum(train['raw_address'].apply(lambda x: '"'in x)))
print(',:',np.sum(train['raw_address'].apply(lambda x: ','in x)))
print('::',np.sum(train['raw_address'].apply(lambda x: ':'in x)))


# In[12]:


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 0
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(),
                                                    s["tag"].values.tolist())]
        self.grouped = self.data.groupby("id").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["id: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


# In[13]:


getter = SentenceGetter(train_ez_transformed)


# In[14]:


sentences = [[word[0] for word in sentence] for sentence in getter.sentences]


# In[15]:


labels = [[s[1] for s in sentence] for sentence in getter.sentences]


# In[16]:


tag_values = list(set(train_ez_transformed["tag"].values))
tag_values.append("PAD")
tag2idx = {t: i for i, t in enumerate(tag_values)}


# In[55]:


from transformers import BertTokenizer, BertModel
model_name='cahya/bert-base-indonesian-522M'
tokenizer = BertTokenizer.from_pretrained(model_name)


# In[18]:


def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        try:
            tokenized_word = tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)
        except:
            pass

    return tokenized_sentence, labels


# In[19]:


tokenized_texts_and_labels = [
    tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(sentences, labels)
]


# In[20]:


tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]


# In[23]:


MAX_LEN = 50
bs = 32


# In[24]:


from tensorflow.keras.preprocessing.sequence import pad_sequences 
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")


# In[26]:


tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")


# In[28]:


attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]


# In[30]:


from sklearn.model_selection import train_test_split
tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)


# In[34]:


tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)


# In[36]:


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)
valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)


# In[55]:


model = BertForTokenClassification.from_pretrained(model_name,
    num_labels=len(tag2idx),
    output_attentions = False,
    output_hidden_states = False
)


# In[43]:


from transformers import BertForTokenClassification, AdamW
FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)


# In[44]:


from transformers import get_linear_schedule_with_warmup

epochs = 3
max_grad_norm = 1.0

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)


# In[52]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


from sklearn.metrics import f1_score, accuracy_score
from tqdm import trange
## Store the average loss after each epoch so we can plot them.
loss_values, validation_loss_values = [], []

for _ in trange(epochs, desc="Epoch"):
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.

    # Put the model into training mode.
    model.train()
    # Reset the total loss for this epoch.
    total_loss = 0

    # Training loop
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # Always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()
        # forward pass
        # This will return the loss (rather than the model output)
        # because we have provided the `labels`.
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        # get the loss
        loss = outputs[0]
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # track train loss
        total_loss += loss.item()
        # Clip the norm of the gradient
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)


    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
        # Move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

    eval_loss = eval_loss / len(valid_dataloader)
    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]
    print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
    print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
    print()


# In[ ]:





# In[ ]:


###FOR EXAMINATION###


# In[29]:


from transformers import BertTokenizer, BertModel
model_name='cahya/bert-base-indonesian-522M'
tokenizer = BertTokenizer.from_pretrained(model_name)


# In[43]:


model = torch.load('/Users/liu/Downloads/model_aftercomma',map_location=torch.device('cpu'))
model_2 = torch.load('/Users/liu/Downloads/model_aftercomma_continue',map_location=torch.device('cpu'))


# In[31]:


tag_values = ['B-s', 'I-P', 'I-s', 'B-P', 'O', 'PAD']


# In[44]:


def get_prediction(raw_address):
  tokenized_sentence = tokenizer.encode(raw_address)
  input_ids = torch.tensor([tokenized_sentence])
  with torch.no_grad():
    output = model(input_ids)
  label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
  tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
  new_tokens, new_labels = [], []
  for token, label_idx in zip(tokens, label_indices[0]):
      if token.startswith("##"):
          new_tokens[-1] = new_tokens[-1] + token[2:]
      else:
          new_labels.append(tag_values[label_idx])
          new_tokens.append(token)
  index_p = [i for i, e in enumerate(new_labels) if (e == 'B-P')| (e =='I-P')]
  index_s = [i for i, e in enumerate(new_labels) if (e == 'B-s')| (e =='I-s')]
  POI = " ".join([new_tokens[ind] for ind in index_p])
  street = " ".join([new_tokens[ind] for ind in index_s])
  return POI, street


# In[139]:


raw_address = comparison.iloc[703]['raw_address_comma']
print(raw_address)


# In[205]:


print(get_prediction(raw_address))
print(get_prediction_2(raw_address))


# In[367]:


def get_prediction_2(raw_address):
#     global tokenized_sentence
#     global input_ids
#     global output
#     global label_indices
#     global tokens
#     global new_tokens
#     global new_labels
#     global POI_ind
#     global street_ind
#     global s_start
#     global s_end

    tokenized_sentence = tokenizer.encode(raw_address)
    input_ids = torch.tensor([tokenized_sentence])
    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)
    POI_ind = [i for i, e in enumerate(new_labels) if (e == 'B-P')| (e =='I-P')]
    street_ind = [i for i, e in enumerate(new_labels) if (e == 'B-s')| (e =='I-s')]
  index_p = [i for i, e in enumerate(new_labels) if (e == 'B-P')| (e =='I-P')]
  index_s = [i for i, e in enumerate(new_labels) if (e == 'B-s')| (e =='I-s')]
  POI = " ".join([new_tokens[ind] for ind in index_p])
  street = " ".join([new_tokens[ind] for ind in index_s])
    return POI, street


# In[ ]:


def get_prediction_3(raw_address):
#     global tokenized_sentence
#     global input_ids
#     global output
#     global label_indices
#     global tokens
#     global new_tokens
#     global new_labels
#     global POI_ind
#     global street_ind
#     global s_start
#     global s_end

    tokenized_sentence = tokenizer.encode(raw_address)
    input_ids = torch.tensor([tokenized_sentence])
    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)
    POI_ind = [i for i, e in enumerate(new_labels) if (e == 'B-P')| (e =='I-P')]
    street_ind = [i for i, e in enumerate(new_labels) if (e == 'B-s')| (e =='I-s')]
    if len(POI_ind) == 0:
        POI = ''
    elif len(POI_ind) == 1:
        POI = raw_address[raw_address.find(new_tokens[POI_ind[0]]):raw_address.find(new_tokens[POI_ind[0]])+len(new_tokens[POI_ind[0]])]
    else:
        p_start = min(POI_ind)
        p_end = max(POI_ind)
        POI = raw_address[raw_address.find(new_tokens[p_start]):raw_address.find(new_tokens[p_end])+len(new_tokens[p_end])]
    if len(street_ind) == 0:
        street = ''
    elif len(street_ind) == 1:
        street = raw_address[raw_address.find(new_tokens[street_ind[0]]):raw_address.find(new_tokens[street_ind[0]])+len(new_tokens[street_ind[0]])]
    else:
        s_start = min(street_ind)
        s_end = max(street_ind)
        street = raw_address[raw_address.find(new_tokens[s_start]):raw_address.find(new_tokens[s_end],raw_address.find(new_tokens[s_start]))+len(new_tokens[s_end])]
    return POI, street


# In[361]:


raw_address = 'ridho bakery, mare raya,'
get_prediction_2(raw_address)


# In[196]:


train = train_ez
comparison = pd.DataFrame()


# In[134]:


train_transformed = pd.read_csv('/Users/liu/train_ez_transformed.csv')


# In[198]:


comparison['id'], comparison['raw_address'], comparison['POI/street']  = train.iloc[:1000]['id'], train.iloc[:1000]['raw_address'], train.iloc[:1000]['POI/street']


# In[199]:


comparison['Prediction1'] = '/'
comparison['Prediction2'] = '/'
comparison = comparison.reset_index(drop=True)
comparison = comparison.fillna('/')


# In[266]:


start = time.time()
for i in range(comparison.shape[0]):
    comparison.at[i,'Prediction1'] = '/'.join(get_prediction(comparison.at[i,'raw_address']))
print('time: ',time.time()-start)


# In[348]:


start = time.time()
for i in range(comparison.shape[0]):
    comparison.at[i,'Prediction2'] = '/'.join(get_prediction_2(comparison.at[i,'raw_address']))
print('time: ',time.time()-start)


# In[360]:


comparison[comparison['POI/street'] != comparison['Prediction2']]


# In[285]:


#法二對 法一錯
comparison[(comparison['POI/street'] == comparison['Prediction2'])&(comparison['POI/street'] != comparison['Prediction1'])]


# In[353]:


#法二錯 法一對
comparison[(comparison['POI/street'] != comparison['Prediction2'])&(comparison['POI/street'] == comparison['Prediction1'])]


# In[288]:


#都錯
comparison[(comparison['POI/street'] != comparison['Prediction2'])&(comparison['POI/street'] != comparison['Prediction1'])].iloc[50:100]


# In[128]:


start = time.time()
for i in range(comparison.shape[0]):
    comparison.at[i,'Prediction3'] = '/'.join(get_prediction(comparison.at[i,'raw_address_comma']))
print('time: ',time.time()-start)


# In[269]:


def replace(x):
    x = x.replace(' .','.')
    x = x.replace(' ,',' ')
    x = x.replace(', ',' ')
    x = x.replace(',','')
    return x


# In[84]:


def replace_comma(x):
    return x.replace(',','')
comparison['raw_address_comma'] = comparison['raw_address'].apply(replace_comma)


# In[350]:


comparison['Prediction1'] = comparison['Prediction1'].apply(replace)
comparison['Prediction2'] = comparison['Prediction2'].apply(replace)


# In[129]:


comparison[~(comparison['Prediction1'] == comparison['POI/street'])].iloc[300:350]


# In[175]:


comparison[(comparison['Prediction1']!='/')&(comparison['POI/street']=='/')]


# In[176]:


comparison[(comparison['Prediction1']!='/')&(comparison['POI/street']=='/')]


# In[368]:


#predict for the whole test set
start = time.time()
for i in range(test.shape[0]):
  test.at[i,'POI/street'] = '/'.join(get_prediction_2(test.iloc[i]['raw_address']))
print('time: ',time.time()-start)
test['POI/street'] = test['POI/street'].apply(replace)


# In[369]:


#output final csv
test.drop(columns='raw_address').to_csv('address_extraction_method2.csv',index=False)


# In[15]:


answer.to_csv('answer_0318noon.csv',index=False)


# In[371]:


accu57 = pd.read_csv('/Users/liu/drop_dup.csv')


# In[379]:


comp = pd.concat([test,accu57],axis=1).drop(columns='id')


# In[381]:


comp.columns = ['raw_address','POI/street_method2','POI/street_accu57']


# In[383]:


comp[comp['POI/street_method2'] != comp['POI/street_accu57']]


# In[15]:


prediction = pd.read_csv('/Users/liu/drop_dup.csv')


# In[45]:


np.sum(prediction['POI/street'].apply(lambda x: 'indu' in x))


# In[46]:


train_hard


# In[385]:


train = pd.read_csv('/Users/liu/Downloads/Address Elements Extraction Dataset/train.csv')


# In[ ]:


#predict for the whole test set
start = time.time()
train['prediction'] = '/'
for i in range(train.shape[0]):
  train.at[i,'prediction'] = '/'.join(get_prediction(train.iloc[i]['raw_address']))
print('time: ',time.time()-start)
train['POI/street'] = train['POI/street'].apply(replace)
train.to_csv('prediction_for_trainingset.csv')


# In[8]:


from nltk.tokenize import word_tokenize
def find_first_index(ra_split, st_split):

    if len(st_split) <= len(ra_split): # new
    
        num_iter = len(ra_split) - len(st_split) + 1
        overlap_list = []

        for i in range(num_iter):
            window = ra_split[i: i+len(st_split)]
            overlap = list(set(window) & set(st_split))
            overlap_list.append(len(overlap))

        max_overlap = [e for e in range(len(overlap_list)) if overlap_list[e] == max(overlap_list)]
        if len(max_overlap) == 1:
            return max_overlap[0]

        else:
            count_list = []
            for idx in max_overlap:
                subset_ra = ra_split[idx: idx+len(st_split)]
                count = 0
                for e in range(len(subset_ra)):
                    if subset_ra[e] not in st_split[e]:
                        count += 0
                    else:
                        count += 1
                count_list.append(count)
            index = count_list.index(max(count_list))
            return max_overlap[index]
    else:
        return 0


# In[9]:


def fix_street_errors(row):
    global raw_add_split
    global extr_street_split
    global index_in_ra
    raw_add = row['raw_address']

    # If a street name is extracted...
    if row['POI'] != "":

        raw_add_split = word_tokenize(raw_add)
        
        extr_street = row['POI']
        extr_street_split = word_tokenize(extr_street)
        
        # If the extracted street is in the raw address as an entire string, good!
        if extr_street in raw_add:
            return raw_add
        
        # This is where there are discrepancies!
        else:
            index_in_ra = find_first_index(raw_add_split, extr_street_split)
            raw_add_split[index_in_ra: index_in_ra+len(extr_street_split)] = extr_street_split
            updated_raw_add = ' '.join(raw_add_split).replace(' ,', ',').replace(' .', '.').replace(' )', ')').replace(' (', '(').replace(' ?', '?')          
            return updated_raw_add
      
    # If a street name is originally an empty string, we just assume there's no error. 
    else:
        return raw_add


# In[10]:


def get_street_mapping_dict(row):
    
    raw_add = row['raw_address']

    # If a street name is extracted...
    if row['street'] != "":

        raw_add_split = word_tokenize(raw_add)
        
        extr_street = row['street']
        extr_street_split = word_tokenize(extr_street)
        
        # If the extracted street is in the raw address as an entire string, good!
        if extr_street in raw_add:
            return None
        
        # This is where there are discrepancies!
        else:
            index_in_ra = find_first_index(raw_add_split, extr_street_split)
            before = raw_add_split[index_in_ra: index_in_ra+len(extr_street_split)] 
            before = ' '.join(before)
            return before, extr_street
      
    # If a street name is originally an empty string, we just assume there's no error. 
    else:
        return None


# In[11]:


start = time.time()

train['street_mapping'] = train.apply(get_street_mapping_dict, axis=1)

print("Executed in {} minutes.".format(round((time.time() - start)/60, 3)))

# Sanity checks
train.loc[[69, 86, 117, 130, 135, 169], :]


# In[12]:


# Create a separate dataframe containing only values in `street_mapping` columns, i.e., rows where changes occurred
df_with_street_mappings = train[train['street_mapping'].notnull()]

street_mapping_dict = dict()

# Create a mapping dictionary, where key is the truncated word/words and the value is the corresponding correct word/words
for row, col in df_with_street_mappings.iterrows():
    street_mapping_dict[col['street_mapping'][0]] = col['street_mapping'][1]
    
# How many street errors are there altogether?
print("Length of street mapping dictionary:", len(street_mapping_dict))


# In[13]:


def get_poi_mapping_dict(row):
    
    raw_add = row['raw_address']

    # If a POI name is extracted...
    if row['POI'] != "":

        raw_add_split = word_tokenize(raw_add)
        
        extr_poi = row['POI']
        extr_poi_split = word_tokenize(extr_poi)
        
        # If the extracted POI is in the raw address as an entire string, good!
        if extr_poi in raw_add:
            return None
        
        # This is where there are discrepancies!
        else:
            index_in_ra = find_first_index(raw_add_split, extr_poi_split)
            before = raw_add_split[index_in_ra: index_in_ra+len(extr_poi_split)] 
            before = ' '.join(before)
            return before, extr_poi
      
    # If a POI name is originally an empty string, we just assume there's no error. 
    else:
        return None


# In[14]:


start = time.time()

train['poi_mapping'] = train.apply(get_poi_mapping_dict, axis=1)

print("Executed in {} minutes.".format(round((time.time() - start)/60, 3)))

# Sanity checks
train.loc[[10, 11, 40, 110, 152, 157, 169], :]


# In[15]:


# Create a separate dataframe containing only values in `poi_mapping` columns, i.e., rows where changes occurred
df_with_poi_mappings = train[train['poi_mapping'].notnull()]

poi_mapping_dict = dict()

# Create a mapping dictionary, where key is the truncated word/words and the value is the corresponding correct word/words
for row, col in df_with_poi_mappings.iterrows():
    poi_mapping_dict[col['poi_mapping'][0]] = col['poi_mapping'][1]
    
# How many POI errors are there altogether?
print("Length of POI mapping dictionary:", len(poi_mapping_dict))


# In[16]:


new_street_mapping_dict = {k: v for k, v in street_mapping_dict.items() if len(k.split()) > 1}
new_poi_mapping_dict = {k: v for k, v in poi_mapping_dict.items() if len(k.split()) > 1}

print("Length of street mapping dictionary after removing single words:", len(new_street_mapping_dict))
print("Length of POI mapping dictionary after removing single words:", len(new_poi_mapping_dict))


# In[17]:


prediction57 = pd.read_csv('/Users/liu/drop_dup.csv')


# In[21]:


pd.DataFrame(['s. par','g. par']).replace({'. ':' '},regex=True)


# In[28]:


replacement = new_street_mapping_dict.copy()
replacement.update(new_poi_mapping_dict)


# In[32]:


s = time.time()
prediction57['fixed'] = prediction57['POI/street'].replace(replacement,regex=True)
print('time: ',(s-time.time())/60)


# In[ ]:


prediction57.to_csv('prediction_fixed_singleno.csv',index=False)


# In[ ]:


new_street_mapping_dict = {k: v for k, v in street_mapping_dict.items() if len(k.split())}
new_poi_mapping_dict = {k: v for k, v in poi_mapping_dict.items() if len(k.split())}

print("Length of street mapping dictionary after removing single words:", len(new_street_mapping_dict))
print("Length of POI mapping dictionary after removing single words:", len(new_poi_mapping_dict))
replacement_2 = new_street_mapping_dict.copy()
replacement_2.update(new_poi_mapping_dict)
print(len(replacement_2))


# In[ ]:


s = time.time()
prediction57['fixed'] = prediction57['POI/street'].replace(replacement_2,regex=True)
print('time: ',(s-time.time())/60)
prediction57.to_csv('prediction_fixed_singleyes.csv',index=False)


# In[189]:


prediction57.head()


# In[192]:


prediction57.loc[0,'POI/street'].replace()


# In[ ]:


for i in range(prediction57.shape[0]):
    for ori, new in replacement.items():
        try:
            prediction57.loc[i,'fixed'] = prediction57.loc[i,'POI/street'].replace(ori,new)
        except:
            pass


# In[ ]:


prediction57.to_csv('prediction_fixed.csv',index=False)


# In[ ]:


replacement_2 = new_street_mapping_dict.copy().update(new_poi_mapping_dict)


# In[ ]:


for i in range(prediction57.shape[0]):
    for ori, new in replacement_2.items():
        try:
            prediction57.loc[i,'fixed'] = prediction57.loc[i,'POI/street'].replace(ori,new)
        except:
            pass


# In[ ]:


prediction57.to_csv('prediction_fixed_nosing.csv',index=False)

