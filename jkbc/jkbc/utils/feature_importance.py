from fastai.basics import *
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
import rfpimp

def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):
    # stolen from fastai.old
    # https://github.com/fastai/fastai/blob/master/old/fastai/structured.py
    if not ignore_flds: ignore_flds=[]
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    else: df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if preproc_fn: preproc_fn(df)
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = pd.Categorical(df[y_fld]).codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
    if do_scale: mapper = scale_vars(df, mapper)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale: res = res + [mapper]
    return res

def fix_missing(df, col, name, na_dict):
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict

def numericalize(df, col, name, max_n_cat):
    if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):
        df[name] = pd.Categorical(col).codes+1
        
def set_rf_samples(n):
    from sklearn.ensemble import forest
    forest._generate_sample_indices = (lambda rs, n_samples:
    forest.check_random_state(rs).randint(0, n_samples, n))

def load_data(path, measure):
    df_raw = pd.read_csv(path)
    df_trn, y_trn, nas = proc_df(df_raw, measure)
    return df_raw, df_trn, y_trn

def convert_data(df_raw, df_trn, y_trn, train_pct = .8):
    def split_vals(data,split): 
        return data[:split], data[split:]
    n_trn = int(len(df_raw)*train_pct)
    X_train, X_valid = split_vals(df_trn, n_trn)
    y_train, y_valid = split_vals(y_trn, n_trn)
    raw_train, raw_valid = split_vals(df_raw, n_trn)
    
    return (X_train, X_valid), (y_train, y_valid), (raw_train, raw_valid)

def get_score(m, X_train, y_train, X_valid, y_valid):
    def rmse(x,y): return math.sqrt(((x-y)**2).mean())
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    return res

def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_, 'std':np.std([tree.feature_importances_ for tree in m.estimators_],
             axis=0)}
                       ).sort_values('imp', ascending=False)

def plot_fi(fi,std=True, feature_importance_type=''):
    if std:
        ax = fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False, xerr='std')
    else:
        ax = fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
    
    ax.set_xlabel(f"{feature_importance_type} Feature Importance")
    return ax

def calculate_feature_importance(model, X_valid, y_valid, use_permutation, max_sample_size = 1000):
    def use_subset(X_valid, y_valid, max_sample_size):
        sample_idx = np.random.permutation(len(X_valid))[:max_sample_size]

        X_valid_sample = X_valid.iloc[sample_idx].copy()
        y_valid_sample = y_valid[sample_idx].copy()
        return X_valid_sample, y_valid_sample
    
    X_valid_sample, y_valid_sample = use_subset(X_valid, y_valid, max_sample_size)

    if not use_permutation:
        return rf_feat_importance(model, X_valid_sample)
    
    fi_permutation = rfpimp.importances(model, X_valid_sample, y_valid_sample) # permutation
    fi_permutation['Importance'] = fi_permutation['Importance']/ fi_permutation['Importance'].sum()
    
    return (fi_permutation
                  .reset_index()
                  .rename({'Feature':'cols', 'Importance':'imp'},axis=1))

def train_regressor(X_train, y_train, n_estimators=40, min_samples_leaf=3, max_features=0.5):
    m = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features, n_jobs=-1, oob_score=True)
    m.fit(X_train, y_train)
    return m