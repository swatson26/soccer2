import pandas as pd
import numpy as np
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import pickle
import db_utils
from sqlalchemy.engine import create_engine
import prettytable
from StringIO import StringIO
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import precision_recall_curve
from matplotlib import cm
from matplotlib.colors import rgb2hex


ALL_LEAGUES = ['Premier', 'Championship', 'SerieA',
               'laLiga1', 'Bundesliga1', 'Championnat']
SEASON_SPLITS = {'train_test':10,'validate':6}


def pprint_df(df):
    output = StringIO()
    df.to_csv(output)
    output.seek(0)
    pt = prettytable.from_csv(output)
    print pt


class TrainTestValidate:
    def __init__(self,user,db_pwd,train_stop_date,drop_na=False):
        self.drop_na = drop_na
        self.train_stop_date = train_stop_date
        eng = "postgresql+psycopg2://%s:%s@localhost:5432/soccer"%(user,db_pwd)
        engine = create_engine(eng)
        conn = engine.connect()
        self._engine = engine
        self._conn = conn
        self.run()

    def _get_seasons(self):
        SQL = """
        SELECT * FROM seasons
        """
        seasons = pd.read_sql(sql=SQL, con=self._engine)
        order = seasons['season'].str[0:4].sort_values().index
        seasons = seasons.reindex(order)
        train_test_seasons = seasons[0:SEASON_SPLITS['train_test']]
        validate_seasons = seasons[SEASON_SPLITS['train_test']:SEASON_SPLITS['train_test']+SEASON_SPLITS['validate']]
        return train_test_seasons, validate_seasons

    def _get_model_inputs(self):
        SQL = """
        SELECT * FROM model_inputs

        """
        model_input_df = pd.read_sql(sql=SQL, con=self._engine)
        model_input_df.index = model_input_df['game_id']
        return model_input_df

    def _get_games(self):
        SQL = """
        SELECT * FROM games
        """
        games_df = pd.read_sql(sql=SQL, con=self._engine)
        return games_df

    def _create_train_test(self):
        train_test_seasons, validate_seasons = self._get_seasons()
        full_games_df = self._get_games()
        model_input_df = self._get_model_inputs()
        games_df = full_games_df.loc[:,['game_id','league','season','result','game_date']]
        full_df = pd.merge(games_df, model_input_df,on=['game_id'])
        full_df.index = full_df['game_id']
        full_df = full_df[full_df['season'].isin(train_test_seasons['season'])]
        full_df['game_date'] = pd.to_datetime(full_df['game_date'])
        full_df.index = full_df['game_date']
        if self.drop_na is True:
            full_df = full_df.dropna()

        train_test_data = {}
        for l in games_df['league'].unique():
            df = full_df[full_df['league']==l]
            train = df[:self.train_stop_date]
            test = df[self.train_stop_date:]

            features = list(model_input_df.columns)
            features = [i for i in features if i not in ['home_team_name','away_team_name']]

            X_train_df = train.loc[:,features]
            y_train = train['result']

            X_test_df = test.loc[:,features]
            y_test = test['result']

            train_test_data[l] = {'train_df':train,
                                  'test_df':test,
                                  'X_train_df':X_train_df,
                                  'y_train':y_train,
                                  'X_test_df':X_test_df,
                                  'y_test':y_test,
                                  'games_df':full_games_df[full_games_df['league']==l]
                                 }
        self.train_test_data = train_test_data

    def run(self):
        self._create_train_test()


class XgbFlow:
    def __init__(self,TrainTestValidate,league,print_output):
        self.tt = TrainTestValidate
        self.league = league
        self._print_output = print_output
        self.initial_model = None
        self._run_intialize()

    def _convert_to_binary_class(self):
         for ix,data in self.tt.train_test_data.iteritems():
                data['y_train'] = np.where(data['y_train']==0,1,0)
                data['y_test'] = np.where(data['y_test']==0,1,0)


    def _gridsearch(self):
        for ix,data in self.tt.train_test_data.iteritems():
            if ix==self.league:
                X_train = data['X_train_df'].drop('game_id',axis=1)
                y_train = data['y_train']
                params = {'max_depth': [3,5,7],
                          'min_child_weight': [3,5,7],
                          'learning_rate': [0.01,0.001],
                          'colsample_bytree': [0.5,0.7],
                          'subsample': [0.4,0.6,0.8],
                          'max_delta_step':[0,1]
                         }
                ind_params = {'n_estimators': 4000,
                              'seed':26,
                              'early_stopping_rounds':200,
                              'objective':'binary:logistic'
                             }
                xgb_model = GridSearchCV(xgb.XGBClassifier(ind_params),
                                         params,
                                         scoring='neg_log_loss',
                                         cv=3,
                                         n_jobs=-1)
                initial_xgb_model = xgb_model.fit(X_train, y_train)
                y = initial_xgb_model.best_estimator_.feature_importances_
                x = X_train.columns
                feat_imp_df = pd.DataFrame({'features':x,'values':y})
                feat_imp_df = feat_imp_df.sort_values('values')
                self.initial_model = {'best_insample_params':initial_xgb_model.best_params_,
                                      'best_insample_score':initial_xgb_model.best_score_,
                                      'insample_feat_imp_df':feat_imp_df,
                                      'insample_result_df':np.nan}


    def train_n_predict(self,params,train_df,y_train,test_df,y_test):
        xgb_matrix_train = xgb.DMatrix(train_df,
                                       label=y_train,
                                       missing = np.nan
                                      )
        xgb_matrix_test = xgb.DMatrix(test_df,
                                      label=y_test,
                                      missing = np.nan
                                     )
        final_params = {'eta':params['learning_rate'],
                        'seed':26,
                        "silent":1,
                        'subsample': params['subsample'],
                        'colsample_bytree': params['colsample_bytree'],
                        'objective': 'binary:logistic',
                        'max_depth':params['max_depth'],
                        'min_child_weight':params['min_child_weight'],
                        'eval_metric':'logloss'}

        trained_xgb_model = xgb.train(params=final_params,
                                      verbose_eval=False,
                                      dtrain=xgb_matrix_train,
                                      num_boost_round=6000,
                                      early_stopping_rounds=800,
                                      evals=[(xgb_matrix_train,'train'),
                                              (xgb_matrix_test,'eval')]
                                      )
        y_scores = trained_xgb_model.predict(xgb_matrix_test)
        return y_scores

    def run_model(self,max_date):
        data = self.tt.train_test_data[self.league]
        games_df = self.tt.train_test_data[self.league]['games_df']
        params = self.initial_model['best_insample_params']
        train_df = data['X_train_df']
        y_train = data['y_train']
        test_df = data['X_test_df']
        y_test = data['y_test']
        y_test_df = pd.DataFrame({'y_test':y_test,'game_date':test_df.index})
        y_train_df = pd.DataFrame({'y_train':y_train,'game_date':train_df.index})
        y_train_df.index = y_train_df['game_date']
        y_test_df.index = y_test_df['game_date']
        dates = test_df[:max_date].index.unique()
        game_ids = []
        y_scores = []
        for d in dates:
            game_ids_ = test_df[d:d]['game_id']
            train_df_ = train_df[:d].drop('game_id',axis=1)
            test_df_ = test_df[d:d].drop('game_id',axis=1)
            y_train_ = y_train_df[:d]
            y_test_ = y_test_df[d:d]
            y_scores_ = self.train_n_predict(params=params,
                                             y_train=y_train_['y_train'],
                                             y_test=y_test_['y_test'],
                                             train_df=train_df_,
                                             test_df=test_df_)
            y_scores.extend(y_scores_)
            game_ids.extend(game_ids_)
            out = games_df[games_df['game_id'].isin(game_ids_)].loc[:,['home_team_name',
                                                                       'away_team_name',
                                                                       'home_goals_scored',
                                                                       'away_goals_scored',
                                                                       'home_avg_win_odds',
                                                                       'draw_avg_win_odds']
                                                                   ]
            y_scores_round = [ round(elem, 2) for elem in y_scores_]
            out.index = out['home_team_name']
            out['score'] = out['home_goals_scored'].astype(str) + ' | ' + out['away_goals_scored'].astype(str)
            out['y_scores'] = y_scores_round
            out = out.drop(['home_team_name','home_goals_scored','away_goals_scored'],axis=1)
            if self._print_output:
                print d.strftime('%Y-%m-%d')
                print ''
                pprint_df(out)
                print ''
                print ''
                print ''
                print '------------------'
        result_df = pd.DataFrame({'game_id':game_ids,
                                  'y_score':y_scores})
        result_df = pd.merge(result_df,games_df.loc[:,['game_id','result','home_avg_win_odds','game_date']],
                             on='game_id')
        result_df['home_win'] = np.where(result_df['result']==0,1,0)
        self.result_df = result_df

    def _run_intialize(self):
        print 'finding initial model paramaters'
        self._convert_to_binary_class()
        self._gridsearch()


class Eval:
    """
    class for evaluating a list of model objects

    Attributes:
    models_list (list): a list of dictionaries that take the form \
    {'model':model_obj,'model_id':some_string}

    """
    def __init__(self,models_list):
        self.models_list = models_list


    def print_model_params(self):
        for m in self.models_list:
            print '--------------'
            print m['league']
            print m['model'].initial_model['best_insample_params']


    def plot_pr_curves(self):
        for m in self.models_list:
            lg = m['league']
            mdl = m['model']
            self._plot_pr_curve(lg,mdl)


    def plot_mean_std_returns(self,n=200):
        for m in self.models_list:
            lg = m['league']
            mdl = m['model']
            result2_df = mdl.result_df.copy()
            mean_rtns = []
            mean_rtns_kelly = []
            thresholds = []
            stds = []
            stds_kelly = []
            n=n
            for ix,i in enumerate(np.linspace(0.0,0.9,n)):
                wagers = []
                for ix, row in result2_df.iterrows():
                    wagers.append(kelly_criterion(row['home_avg_win_odds'],row['y_score'],i))
                result2_df['kelly_wager'] = wagers
                result2_df['ex_rtn_kelly'] = np.where((result2_df['home_win']==1)&(result2_df['kelly_wager']>0),
                                                  (result2_df['home_avg_win_odds']-1)*result2_df['kelly_wager'],
                                                   np.where((result2_df['home_win']!=1)&(result2_df['kelly_wager']>0),
                                                            -result2_df['kelly_wager'],0)
                                           )
                result2_df['ex_rtn'] = np.where((result2_df['home_win']==1)&(result2_df['y_score']>i),
                                             result2_df['home_avg_win_odds']-1,
                                            np.where((result2_df['home_win']!=1)&(result2_df['y_score']>i),
                                                     -1,0)
                                           )
                result2_filter = result2_df[result2_df['ex_rtn_kelly']!=0]
                result2_df.index = result2_df['game_date']
                thresholds.append(round(i,2))
                mean_rtns.append(result2_filter['ex_rtn'].mean())
                stds.append(result2_filter['ex_rtn'].std())
                mean_rtns_kelly.append(result2_filter['ex_rtn_kelly'].mean())
                stds_kelly.append(result2_filter['ex_rtn_kelly'].std())
            plt_kelly_df = pd.DataFrame({'t':thresholds,
                                         'rtn':mean_rtns_kelly,
                                         'std':stds_kelly
                                        })
            plt_df = pd.DataFrame({'t':thresholds,
                                   'rtn':mean_rtns,
                                   'std':stds
                                  })
            plt_df = plt_df.sort_values('t',ascending=False)
            data = [
                go.Scatter(
                    x=plt_df['std'],
                    y=plt_df['rtn'],
                    text=plt_df['t'],
                    name='Naive Allocation',
                    mode='markers+lines',
                    marker=dict(
                    color = plt_df['t'],
                    colorscale='Viridis',
                    size=10,
                        line = dict(
                        color = 'black',
                        width = 2),
                    showscale=True),
                    line=dict(
                    color = 'gray',
                    width=2)
                ),
                    go.Scatter(
                    x=plt_kelly_df['std'],
                    y=plt_kelly_df['rtn'],
                    text=plt_kelly_df['t'],
                    name='Kelly Criterion',
                    mode='markers+lines',
                    marker=dict(
                    color = plt_kelly_df['t'],
                    colorscale='Viridis',
                    size=10,
                        line = dict(
                        color = 'black',
                        width = 2),
                    showscale=True),
                    line=dict(
                    color = 'gray',
                    width=2)
                )
            ]
            layout=go.Layout(
                            legend=dict(orientation="h"),
                            title='%s σ,μ Returns'%(lg),
                            xaxis=dict(
                                title='Standard Deviation',
                                titlefont=dict(
                                    size=18,
                                    color='#7f7f7f'
                                )
                            ),
                            yaxis=dict(
                                title='Mean Return',
                                titlefont=dict(
                                    size=18,
                                    color='#7f7f7f'
                                )
                            )
            )
            fig=go.Figure(data=data,layout=layout)
            iplot(fig)


    def plot_cumulative_profit(self,threshold_start=0.6,threshold_stop=0.9):
        for m in self.models_list:
            lg = m['league']
            mdl = m['model']
            result2_df = mdl.result_df.copy()
            data = []
            n = 10
            cmap = cm.get_cmap('viridis', n)
            cmap2 = cm.get_cmap('viridis', n)
            fig = tools.make_subplots(rows=1, cols=2,shared_yaxes=True,print_grid=False)
            for ix,i in enumerate(np.linspace(threshold_start,threshold_stop,n)):
                rgb = cmap(ix)[:3]
                hexx = rgb2hex(rgb)
                rgb2 = cmap2(ix)[:3]
                hexx2 = rgb2hex(rgb2)
                wagers = []
                for ix, row in result2_df.iterrows():
                    wagers.append(kelly_criterion(row['home_avg_win_odds'],row['y_score'],i))
                result2_df['kelly_wager'] = wagers
                result2_df['ex_rtn_kelly'] = np.where((result2_df['home_win']==1)&(result2_df['kelly_wager']>0),
                                                  (result2_df['home_avg_win_odds']-1)*result2_df['kelly_wager'],
                                                   np.where((result2_df['home_win']!=1)&(result2_df['kelly_wager']>0),
                                                            -result2_df['kelly_wager'],0)
                                           )
                result2_df['ex_rtn'] = np.where((result2_df['home_win']==1)&(result2_df['y_score']>i),
                                             result2_df['home_avg_win_odds']-1,
                                            np.where((result2_df['home_win']!=1)&(result2_df['y_score']>i),
                                                     -1,0)
                                           )
                result2_df.index = result2_df['game_date']
                trace1 = go.Scatter(
                                showlegend=False,
                                x=result2_df['ex_rtn'].index,
                                y=result2_df['ex_rtn'].cumsum(),
                                name=str(round(i,2)),
                                line = dict(
                                            color = (hexx),
                                            width = 2),

                )
                fig.append_trace(trace1, 1, 1)
                trace2 = go.Scatter(
                    showlegend=False,
                     name=str(round(i,2)),
                    x=result2_df['ex_rtn_kelly'].index,
                    y=result2_df['ex_rtn_kelly'].cumsum(),
                   line = dict(
                                            color = (hexx2),
                                            width = 2),
                )
                fig.append_trace(trace2, 1, 2)
            fig['layout'].update(height=600, width=1000,
                                 title='%s Cumulative Profits <br> for Naive Allocation and Kelly Criterion'%(lg))
            iplot(fig)


    def get_stats(self,threshold=0.85):
        for m in self.models_list:
            lg = m['league']
            mdl = m['model']
            result2_df = mdl.result_df.copy()
            print '--------------------------'
            print lg
            print '--------------------------'
            wagers = []
            for ix, row in result2_df.iterrows():
                wagers.append(self._kelly_criterion(row['home_avg_win_odds'],row['y_score'],threshold))
            result2_df['kelly_wager'] = wagers
            result2_df['ex_rtn_kelly'] = np.where((result2_df['home_win']==1)&(result2_df['kelly_wager']>0),
                                              (result2_df['home_avg_win_odds']-1)*result2_df['kelly_wager'],
                                               np.where((result2_df['home_win']!=1)&(result2_df['kelly_wager']>0),
                                                        -result2_df['kelly_wager'],0)
                                       )
            result2_df['ex_rtn'] = np.where((result2_df['home_win']==1)&(result2_df['y_score']>threshold),
                                         result2_df['home_avg_win_odds']-1,
                                        np.where((result2_df['home_win']!=1)&(result2_df['y_score']>threshold),
                                                 -1,0)
                                       )
            result2_df.index = pd.to_datetime(result2_df['game_date'])
            ff = result2_df[result2_df['kelly_wager']!=0]
            print 'Sortino Ratio: '+  str(round(ff['ex_rtn_kelly'].mean()/ff['ex_rtn_kelly'][ff['ex_rtn_kelly']<0].std(),2))
            t = (result2_df.index.max() - result2_df.index.min()).days/365.
            annualized_rtn = (((result2_df['ex_rtn_kelly'].sum()/1)**(1/float(t)))-1)*100.
            print 'Annualized Returns: %s%%'%(str(round(annualized_rtn,2)))
            a = len(result2_df)
            f = len(ff)
            print 'Betting Rate: %s%%'%(round(float(f)/a*100.,2))
            prec = len(result2_df[(result2_df['kelly_wager']!=0)&(result2_df['home_win']==1)])/float(len(result2_df[result2_df['kelly_wager']!=0]))
            print 'Precision: %s'%(round(prec,3))
            fig, ax = plt.subplots(1,2)
            fig.set_size_inches(16,6)
            xx = ff.groupby(pd.Grouper(freq='w'))['kelly_wager'].sum()[ff.groupby(pd.Grouper(freq='w'))['kelly_wager'].sum()>0]
            sns.distplot(xx,ax=ax[0])
            ax[0].axvline(np.percentile(xx,50),color='orange')
            ax[0].axvline(np.mean(xx),color='green')
            ax[0].set_title('Total Weekly Wager Size \n Median: %s\n Mean: %s'%(str(round(np.mean(xx),3)),
                                                                             str(round(np.median(xx),3))),size=20)
            xx = ff.groupby(pd.Grouper(freq='w'))['ex_rtn_kelly'].sum()[ff.groupby(pd.Grouper(freq='w'))['kelly_wager'].sum()>0]
            sns.distplot(xx,
                         ax=ax[1],
                         norm_hist=True,
                         label='historical')
            ax[1].axvline(np.percentile(xx,1),color='red')
            ax[1].axvline(np.mean(xx),color='green')
            ax[1].axvline(np.median(xx),color='orange')
            ax[1].set_title('Historical Weekly Returns \n 1st Percentile: %s\n Mean: %s\n Median: %s'%(str(round(np.percentile(xx,1),3)),
                                                                                                     str(round(np.mean(xx),3)),
                                                                                                     str(round(np.median(xx),3))
                                                                                                    ),size=20)
            plt.show()


    ## Helper Functions
    def _kelly_criterion(self, win_odds, win_prob,thresh):
        if win_prob >= thresh:
            alloc = ((win_prob*win_odds) - 1)/(float(win_odds)-1)
        else:
            alloc = 0

        if alloc < 0:
            alloc = 0
        return alloc

    def _plot_pr_curve(self,league,model_obj):
        precision, recall, thresholds = precision_recall_curve(model_obj.result_df['home_win'],
                                                               model_obj.result_df['y_score'],)
        yprob = model_obj.result_df['y_score']
        y_actual = model_obj.result_df['home_win']
        response_rates = []
        for b in thresholds:
            rr = str(round(sum(yprob>b)/float(len(y_actual)),3))
            ss = rr + '|' + str(round(b,3))
            response_rates.append(ss)
        data = [
            go.Scatter(
                x=recall,
                y=precision,
                text=response_rates
            )
        ]
        layout=go.Layout(title=league,
                            xaxis=dict(
                            title='recall',
                            titlefont=dict(
                                size=9,
                                color='#7f7f7f'
                            )
                        ),
                        yaxis=dict(
                            title='precision',
                            titlefont=dict(
                                size=9,
                                color='#7f7f7f'
                            )
                        ))
        fig=go.Figure(data=data,layout=layout)
        iplot(fig)
