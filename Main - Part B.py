##import
import matplotlib.pyplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from pandas.plotting import parallel_coordinates
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn import metrics

random_state=123

## new data
data = pd.read_excel(r'C:\Users\10ofi\Desktop\MLproject\CarInsurance_Xy_train_part2.xlsx')
features_before = data.drop(['OUTCOME','ID', 'AGE', 'POSTAL_CODE'], axis=1)
features_after = data.drop(['OUTCOME','ID', 'AGE_BEFORE', 'POSTAL_CODE_BEFORE'], axis=1)
outcome = data['OUTCOME']
features_before_dummy = pd.get_dummies(features_before, drop_first=False) ##convet to dummy
features_after_dummy = pd.get_dummies(features_after, drop_first=False) ##convet to dummy

X_train1, X_test1, Y_train1, Y_test1 = train_test_split(features_before_dummy, outcome, test_size=0.3, random_state=random_state)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(features_after_dummy, outcome, test_size=0.3, random_state=random_state)

print(f"trainSize: {X_train1.shape[0]}")
print(f"testSize: {X_test1.shape[0]}")
print("train",Y_train1.value_counts()/Y_train1.shape[0])
print("test",Y_test1.value_counts()/Y_test1.shape[0])

print(f"trainSize: {X_train2.shape[0]}")
print(f"testSize: {X_test2.shape[0]}")
print("train",Y_train2.value_counts()/Y_train2.shape[0])
print("test",Y_test2.value_counts()/Y_test2.shape[0])

################################### Decision Trees ###################################

## full tree
model1 = DecisionTreeClassifier(criterion='entropy', random_state=random_state)
model1.fit(X_train1, Y_train1)
plt.figure(figsize=(35, 100))
plot_tree(model1, filled=True, class_names=True, feature_names=X_train1.columns.values)
print(f"Accuracy of full tree with train data: {accuracy_score(y_true=Y_train1, y_pred=model1.predict(X_train1)):.3f}")
print(f"Accuracy of full tree with train data: {accuracy_score(y_true=Y_test1, y_pred=model1.predict(X_test1)):.3f}")

model2 = DecisionTreeClassifier(criterion='entropy',random_state=random_state)
model2.fit(X_train2, Y_train2)
plt.figure(figsize=(35, 100))
plot_tree(model2, filled=True, class_names=True, feature_names=X_train2.columns.values)
print(f"Accuracy of full tree with train data: {accuracy_score(y_true=Y_train2, y_pred=model2.predict(X_train2)):.3f}")
print(f"Accuracy of full tree with train data: {accuracy_score(y_true=Y_test2, y_pred=model2.predict(X_test2)):.3f}")
print(model2.get_params(), '\n')

## chose parameter
#### grid_search
param_grid_DT = {'criterion': ['entropy', 'gini', 'log_loss'],
                            'max_depth': np.arange(3, 11, 1),
                            'max_features': np.arange(5, 11, 1)}
param_grid_DT.values()

grid_search_DT = GridSearchCV(estimator=DecisionTreeClassifier(random_state=random_state),
                           param_grid=param_grid_DT,
                           refit=True,
                           cv=10, verbose=3)
grid_search_DT.fit(X_train2, Y_train2)

best_model_DT = grid_search_DT.best_estimator_
print(grid_search_DT.best_params_, '\n')
print(best_model_DT.get_params(), '\n')
gridSearchCombination_DT = pd.DataFrame(grid_search_DT.cv_results_).sort_values(by= ['mean_test_score'], ascending=False)
gridSearchCombination_subDT = gridSearchCombination_DT[['param_criterion','param_max_depth','param_max_features', 'mean_test_score', 'rank_test_score']]

plt.figure(figsize=(30, 40))
plot_tree(best_model_DT, filled=True, class_names=True, feature_names=X_train2.columns.values)
preds_DT = best_model_DT.predict(X_test2)
print("Train accuracy: ",round(max(gridSearchCombination_DT['mean_test_score']),3))
print("Test accuracy: ", round(accuracy_score(Y_test2, preds_DT), 3))


#### hamdany algoritem
X_train2_, X_validation2, Y_train2_, Y_validation2 = train_test_split(X_train2, Y_train2, test_size=0.3, random_state=random_state)

###### criterion
criterion_list =['entropy', 'gini', 'log_loss']
res1 = pd.DataFrame()
for criterion_ in criterion_list:
    model_CR = DecisionTreeClassifier(criterion=criterion_, random_state=random_state)
    model_CR.fit(X_train2_, Y_train2_)
    res1 = res1.append({'criterion':criterion_,
                      'train_acc':accuracy_score(Y_train2_, model_CR.predict(X_train2_)),
                      'validation_acc':accuracy_score(Y_validation2, model_CR.predict(X_validation2)),}, ignore_index=True,)

objects = (res1['criterion'])
y_pos = np.arange(len(objects))
performance = res1['validation_acc']
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('The accuracy of different criterions')
plt.show()

###### max_features
max_features_list = np.arange(1, 31, 1)
res2 = pd.DataFrame()
for max_features in max_features_list:
    model_MF = DecisionTreeClassifier(criterion='entropy', max_features=max_features, random_state=random_state)
    model_MF.fit(X_train2_, Y_train2_)
    res2 = res2.append({'max_features': max_features,
                      'train_acc':accuracy_score(Y_train2_, model_MF.predict(X_train2_)),
                      'validation_acc':accuracy_score(Y_validation2, model_MF.predict(X_validation2)),
                      'criterion':'entropy'}, ignore_index=True,)

    model_MF = DecisionTreeClassifier(criterion='log_loss', max_features=max_features, random_state=random_state)
    model_MF.fit(X_train2_, Y_train2_)
    res2 = res2.append({'max_features': max_features,
                      'train_acc':accuracy_score(Y_train2_, model_MF.predict(X_train2_)),
                      'validation_acc':accuracy_score(Y_validation2, model_MF.predict(X_validation2)),
                      'criterion':'log_loss'}, ignore_index=True,)

res2 = res2.sort_values('validation_acc',ascending=False)
sns.lineplot(data=res2, x=res2['max_features'], y=res2['validation_acc'] , marker='o', markersize=4)
sns.lineplot(data=res2, x=res2['max_features'], y=res2['train_acc'], marker='o', markersize=4)
sns.legend(['validation Accuracy', 'train Accuracy'])
plt.show()

###### max_depth
max_depth_list = np.arange(1, 31, 1)
res3 = pd.DataFrame()
for max_depth in max_depth_list:
    model_MD = DecisionTreeClassifier(criterion='entropy',max_features=12, max_depth=max_depth, random_state=random_state)
    model_MD.fit(X_train2_, Y_train2_)
    res3 = res3.append({'max_depth': max_depth,
                      'train_acc':accuracy_score(Y_train2_, model_MD.predict(X_train2_)),
                      'validation_acc':accuracy_score(Y_validation2, model_MD.predict(X_validation2)),
                      'criterion':'entropy'}, ignore_index=True,)

    model_MD = DecisionTreeClassifier(criterion='entropy',max_features=12, max_depth=max_depth, random_state=random_state)
    model_MD.fit(X_train2_, Y_train2_)
    res3 = res3.append({'max_depth': max_depth,
                      'train_acc':accuracy_score(Y_train2_, model_MD.predict(X_train2_)),
                      'validation_acc':accuracy_score(Y_validation2, model_MD.predict(X_validation2)),
                      'criterion':'log_loss'}, ignore_index=True,)

res3 = res3.sort_values('validation_acc',ascending=False)
sns.lineplot(data=res3, x=res3['max_depth'], y=res3['validation_acc'],marker='o', markersize=4)
sns.lineplot(data=res3, x=res3['max_depth'], y=res3['train_acc'], marker='o', markersize=4)
sns.legend(['validation Accuracy', 'train Accuracy'])
plt.show()

##### accuracy hamdany algoritem
hamdany_DT = {'criterion': ['entropy'],
              'max_depth': [11],
              'max_features': [12] }
hamdany_DT.values()

hamdany_search_DT = GridSearchCV(estimator=DecisionTreeClassifier(random_state=random_state),
                           param_grid=hamdany_DT,
                           refit=True,
                           cv=2, verbose=3)
hamdany_search_DT.fit(X_train2, Y_train2)

model_hamdany = hamdany_search_DT.best_estimator_
print(hamdany_search_DT.best_params_, '\n')
print(hamdany_search_DT.get_params(), '\n')
hamdany_Combination_DT = pd.DataFrame(hamdany_search_DT.cv_results_).sort_values(by= ['mean_test_score'], ascending=False)

plt.figure(figsize=(30, 40))
plot_tree(model_hamdany, filled=True, class_names=True, feature_names=X_train2.columns.values)
preds_DT_hamdany = model_hamdany.predict(X_test2)
print("Train accuracy: ",round(max(hamdany_Combination_DT['mean_test_score']),3))
print("Test accuracy: ", round(accuracy_score(Y_test2, preds_DT_hamdany), 3))


## feature importances
feature_importances = pd.Series(best_model_DT.feature_importances_)
feature_importances.index = [X_test2.columns.values]
feature_importances = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(1, 5))
plt.style.use('ggplot')
feature_importances.T.plot(kind='bar')
plt.xlabel('Features')
plt.ylabel('Feature importances')


################################### Neural Networks  ###################################

X_train_NN = X_train2.drop(['VEHICLE_TYPE_sedan','VEHICLE_TYPE_sports car', 'ANNUAL_MILEAGE'], axis=1)
X_test_NN = X_test2.drop(['VEHICLE_TYPE_sedan','VEHICLE_TYPE_sports car', 'ANNUAL_MILEAGE'], axis=1)
X_train_NN['SPEEDING_VIOLATIONS_NEW'] = np.where(X_train_NN['SPEEDING_VIOLATIONS_NEW'] > 0.5, 0.5, X_train_NN['SPEEDING_VIOLATIONS_NEW']).tolist()

## nirmul
minmax_scaler = MinMaxScaler()
X_train_n = minmax_scaler.fit_transform(X_train_NN)
X_test_n = minmax_scaler.transform(X_test_NN)

## default NN
model_default_NN = MLPClassifier(random_state=random_state)
model_default_NN.fit(X_train_n, Y_train2)
print("Train accuracy: ",round(accuracy_score(Y_train2, model_default_NN.predict(X_train_n)),3))
print("Test accuracy: ", round(accuracy_score(Y_test2, model_default_NN.predict(X_test_n)), 3))

## chose parameter
####  1 hidden layer
param_grid_NN = {'hidden_layer_sizes':  range(1, 100, 2),
                'max_iter': range(1, 501, 10),
                'learning_rate_init': [0.001*lr for lr in range(1, 5, 1)],
                'activation': ['logistic','relu']}
param_grid_NN.values()

grid_search_NN = GridSearchCV(estimator=MLPClassifier(random_state=random_state),
                           param_grid=param_grid_NN,
                           refit=True,scoring='accuracy',
                           cv=5, verbose=3)
grid_search_NN.fit(X_train_n, Y_train2)

best_model_NN = grid_search_NN.best_estimator_
print(grid_search_NN.best_params_, '\n')
print(best_model_NN.get_params(), '\n')
gridSearchCombination_NN = pd.DataFrame(grid_search_NN.cv_results_).sort_values(by= ['rank_test_score'], ascending=True)
gridSearchCombination_sub_NN = gridSearchCombination_NN[['param_hidden_layer_sizes','param_learning_rate_init','param_max_iter','param_activation','mean_test_score']]

preds_NN = best_model_NN.predict(X_test_n)
print("Train accuracy: ",round(max(gridSearchCombination_NN['mean_test_score']),3))
print("Test accuracy: ", round(accuracy_score(Y_test2, preds_NN), 3))


######graph
gridSearchCombination_sub_NN = gridSearchCombination_NN[['param_hidden_layer_sizes','param_learning_rate_init','param_max_iter','param_activation','mean_test_score']].head(50)
gridSearchCombination_sub_NN["param_hidden_layer_sizes"] = 5 * gridSearchCombination_sub_NN["param_hidden_layer_sizes"]
gridSearchCombination_sub_NN["mean_test_score"] = 500 * gridSearchCombination_sub_NN["mean_test_score"]
gridSearchCombination_sub_NN["param_learning_rate_init"] = 100000 * gridSearchCombination_sub_NN["param_learning_rate_init"]
parallel_coordinates(gridSearchCombination_sub_NN, 'param_activation', colormap=plt.get_cmap("Set1"))
plt.show()


#### 2 hidden layers
X_train2_n, X_validation2_n, Y_train2_n, Y_validation2_n = train_test_split(X_train_n, Y_train2, test_size=0.3, random_state=random_state)

res4 = pd.DataFrame()
activations = ['logistic','relu']
for act in activations:
    print(f"activation: {act}")
    for max_iter in range(1,501,2):
        print(f"iter: {max_iter}")
        for size_ in range(1, 101, 2):
            print(f"size: {size_}")
            for lr in range(1, 5, 1):
                print(f"lr: {lr}")
                model_2layers = MLPClassifier(random_state=random_state,
                    hidden_layer_sizes=(size_, size_),
                    max_iter=max_iter,
                    activation=act,
                    verbose=False,
                    learning_rate_init=0.001 * lr,
                    alpha=0.00)
                model_2layers.fit(minmax_scaler.transform(X_train2_n), Y_train2_n)
                res4 = res4.append({
                    'size': size_,
                    'max_iter': max_iter,
                    'activation': act,
                    'lr': lr,
                    'train_acc':model_2layers.score(minmax_scaler.transform(X_train2_n), Y_train2_n),
                    'validation_acc': model_2layers.score(minmax_scaler.transform(X_validation2_n), Y_validation2_n)}, ignore_index=True,)

res4 = res4.sort_values(by='validation_acc',ascending=False)
res4["lr"] = res4["lr"] / 1000

res4_top50 = res4[['size', 'max_iter', 'activation', 'lr', 'validation_acc']].head(50)
res4_top50['size'] = res4_top50['size']*5
res4_top50['lr'] = res4_top50['lr']*100000
res4_top50['validation_acc'] = res4_top50['validation_acc']*500
res4_top50['max_iter'] = res4_top50['max_iter']
parallel_coordinates(res4_top50, 'activation', colormap=plt.get_cmap("Set1"))
plt.show()

model_2layers_NN = MLPClassifier(random_state=random_state,hidden_layer_sizes=(9, 9),
                    max_iter=261,
                    activation='relu',
                    verbose=False,
                    learning_rate_init=0.003,
                    alpha=0.00)
model_2layers_NN.fit(X_train_n, Y_train2)
print("Train accuracy: ",round(accuracy_score(Y_train2, model_2layers_NN.predict(X_train_n)),3))
print("Test accuracy: ", round(accuracy_score(Y_test2, model_2layers_NN.predict(X_test_n)), 3))

################################### K means ###################################

X_train_Kmeans = pd.DataFrame(X_train_n)
X_test_Kmeans = pd.DataFrame(X_test_n)
full_X_Kmeans = pd.concat([X_train_Kmeans,X_test_Kmeans])

##PCA
pca = PCA(n_components=2)
pca.fit(X_train_Kmeans)
Kmeans_pca_2 = pca.transform(X_train_Kmeans)
Kmeans_pca_2 = pd.DataFrame(Kmeans_pca_2, columns=['PC1', 'PC2'])

##K-means
#### 2 clusters

kmeans_2 = KMeans(n_clusters=2, random_state=random_state)
kmeans_2.fit(X_train_Kmeans)
cluster_centers_k2 = kmeans_2.cluster_centers_
predictKmeans = kmeans_2.predict(X_train_Kmeans)
Kmeans_pca_2['k=2'] = predictKmeans
sns.scatterplot(x='PC1', y='PC2', hue='k=2', data=Kmeans_pca_2, palette={0:'blue', 1:'green'}, s=150)
plt.scatter(pca.transform(kmeans_2.cluster_centers_)[:, 0], pca.transform(kmeans_2.cluster_centers_)[:, 1], marker='+', s=100 ,color='red')
plt.title("K means for 2 clusters")
plt.show()
print("Train accuracy: ",round(accuracy_score(Y_train2, kmeans_2.predict(X_train_Kmeans)),3))
print("Test accuracy: ", round(accuracy_score(Y_test2, kmeans_2.predict(X_test_Kmeans)), 3))


#### 10 clusters

iner_list = []
dbi_list = []
sil_list = []
chi_list = []
Kmeans_pca = pca.transform(full_X_Kmeans)
Kmeans_pca = pd.DataFrame(Kmeans_pca, columns=['PC1', 'PC2'])

kmeans = KMeans(n_clusters=3, random_state=random_state)
kmeans.fit(full_X_Kmeans)
cluster_centers_k3 = kmeans.cluster_centers_
predictKmeans = kmeans.predict(full_X_Kmeans)
Kmeans_pca['k=3'] = predictKmeans
iner = kmeans.inertia_
sil = silhouette_score(full_X_Kmeans, predictKmeans)
dbi = davies_bouldin_score(full_X_Kmeans, predictKmeans)
labels = kmeans.labels_
cal = metrics.calinski_harabasz_score(full_X_Kmeans, labels)
dbi_list.append(dbi)
sil_list.append(sil)
iner_list.append(iner)
chi_list.append(cal)

kmeans = KMeans(n_clusters=4, random_state=random_state)
kmeans.fit(full_X_Kmeans)
cluster_centers_k4 = kmeans.cluster_centers_
predictKmeans = kmeans.predict(full_X_Kmeans)
Kmeans_pca['k=4'] = predictKmeans
iner = kmeans.inertia_
sil = silhouette_score(full_X_Kmeans, predictKmeans)
dbi = davies_bouldin_score(full_X_Kmeans, predictKmeans)
labels = kmeans.labels_
cal = metrics.calinski_harabasz_score(full_X_Kmeans, labels)
dbi_list.append(dbi)
sil_list.append(sil)
iner_list.append(iner)
chi_list.append(cal)

kmeans = KMeans(n_clusters=5, random_state=random_state)
kmeans.fit(full_X_Kmeans)
cluster_centers_k5 = kmeans.cluster_centers_
predictKmeans = kmeans.predict(full_X_Kmeans)
Kmeans_pca['k=5'] = predictKmeans
iner = kmeans.inertia_
sil = silhouette_score(full_X_Kmeans, predictKmeans)
dbi = davies_bouldin_score(full_X_Kmeans, predictKmeans)
labels = kmeans.labels_
cal = metrics.calinski_harabasz_score(full_X_Kmeans, labels)
dbi_list.append(dbi)
sil_list.append(sil)
iner_list.append(iner)
chi_list.append(cal)

kmeans = KMeans(n_clusters=6, random_state=random_state)
kmeans.fit(full_X_Kmeans)
cluster_centers_k6 = kmeans.cluster_centers_
predictKmeans = kmeans.predict(full_X_Kmeans)
Kmeans_pca['k=6'] = predictKmeans
iner = kmeans.inertia_
sil = silhouette_score(full_X_Kmeans, predictKmeans)
dbi = davies_bouldin_score(full_X_Kmeans, predictKmeans)
labels = kmeans.labels_
cal = metrics.calinski_harabasz_score(full_X_Kmeans, labels)
dbi_list.append(dbi)
sil_list.append(sil)
iner_list.append(iner)
chi_list.append(cal)

kmeans = KMeans(n_clusters=7, random_state=random_state)
kmeans.fit(full_X_Kmeans)
cluster_centers_k7 = kmeans.cluster_centers_
predictKmeans = kmeans.predict(full_X_Kmeans)
Kmeans_pca['k=7'] = predictKmeans
iner = kmeans.inertia_
sil = silhouette_score(full_X_Kmeans, predictKmeans)
dbi = davies_bouldin_score(full_X_Kmeans, predictKmeans)
labels = kmeans.labels_
cal = metrics.calinski_harabasz_score(full_X_Kmeans, labels)
dbi_list.append(dbi)
sil_list.append(sil)
iner_list.append(iner)
chi_list.append(cal)

kmeans = KMeans(n_clusters=8, random_state=random_state)
kmeans.fit(full_X_Kmeans)
cluster_centers_k8 = kmeans.cluster_centers_
predictKmeans = kmeans.predict(full_X_Kmeans)
Kmeans_pca['k=8'] = predictKmeans
iner = kmeans.inertia_
sil = silhouette_score(full_X_Kmeans, predictKmeans)
dbi = davies_bouldin_score(full_X_Kmeans, predictKmeans)
labels = kmeans.labels_
cal = metrics.calinski_harabasz_score(full_X_Kmeans, labels)
dbi_list.append(dbi)
sil_list.append(sil)
iner_list.append(iner)
chi_list.append(cal)

kmeans = KMeans(n_clusters=9, random_state=random_state)
kmeans.fit(full_X_Kmeans)
cluster_centers_k9 = kmeans.cluster_centers_
predictKmeans = kmeans.predict(full_X_Kmeans)
Kmeans_pca['k=9'] = predictKmeans
iner = kmeans.inertia_
sil = silhouette_score(full_X_Kmeans, predictKmeans)
dbi = davies_bouldin_score(full_X_Kmeans, predictKmeans)
labels = kmeans.labels_
cal = metrics.calinski_harabasz_score(full_X_Kmeans, labels)
dbi_list.append(dbi)
sil_list.append(sil)
iner_list.append(iner)
chi_list.append(cal)

kmeans = KMeans(n_clusters=10, random_state=random_state)
kmeans.fit(full_X_Kmeans)
cluster_centers_k10 = kmeans.cluster_centers_
predictKmeans = kmeans.predict(full_X_Kmeans)
Kmeans_pca['k=10'] = predictKmeans
iner = kmeans.inertia_
sil = silhouette_score(full_X_Kmeans, predictKmeans)
dbi = davies_bouldin_score(full_X_Kmeans, predictKmeans)
labels = kmeans.labels_
cal = metrics.calinski_harabasz_score(full_X_Kmeans, labels)
dbi_list.append(dbi)
sil_list.append(sil)
iner_list.append(iner)
chi_list.append(cal)

plt.plot(range(3, 11, 1), iner_list, marker='o')
plt.title("Inertia")
plt.xlabel("Number of clusters")
plt.show()

plt.plot(range(3, 11, 1), sil_list, marker='o')
plt.title("Silhouette")
plt.xlabel("Number of clusters")
plt.show()

plt.plot(range(3, 11, 1), dbi_list, marker='o')
plt.title("Davies-bouldin")
plt.xlabel("Number of clusters")
plt.show()

plt.plot(range(3, 11, 1), chi_list, marker='o')
plt.title("Calinski-Harabasz Index")
plt.xlabel("Number of clusters")
plt.show()

####real outcome

Kmeans_pca.to_excel(r'C:\Users\10ofi\Desktop\MLfa\KmeansPCA.xlsx', index=False)
Y_train2.to_excel(r'C:\Users\10ofi\Desktop\MLfa\outcomeK.xlsx', index=False)
full_X_Kmeans.to_excel(r'C:\Users\10ofi\Desktop\MLfa\X_trainK.xlsx', index=False)
Kmeans_pca = pd.read_excel(r'C:\Users\10ofi\Desktop\MLfa\input_pycharm.xlsx')

sns.scatterplot(x='PC1', y='PC2', hue='OUTCOME', data=Kmeans_pca, palette={0:'blue', 1:'red'}, s=150)
plt.title("Division of Outcome class's in the training set")
plt.show()


################################### Agglomerative Clustering ###################################

model_AC = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'ward')
clustering_AC = model_AC.fit(full_X_Kmeans)
clustering_AC_predict = clustering_AC.labels_
Kmeans_pca['Agglomerative Clustering'] = clustering_AC_predict
sns.scatterplot(x='PC1', y='PC2', hue='Agglomerative Clustering', data=Kmeans_pca, palette={0:'blue', 1:'green', 2:'red', 3:'black'}, s=200, )
sns.scatterplot(x='PC1', y='PC2', hue='k=4', data=Kmeans_pca, palette={0:'green', 1:'red', 2:'blue', 3:'black'}, s=200)

full_X_Kmeans.to_excel(r'C:\Users\10ofi\Desktop\MLfa\X_full.xlsx', index=False)
Kmeans_pca.to_excel(r'C:\Users\10ofi\Desktop\MLfa\Kmeans_pca_new.xlsx', index=False)


################################### choose modle ###################################

## confusion matrix
CM_DT = confusion_matrix(y_true=Y_test2, y_pred=best_model_DT.predict(X_test2))
CM_NN = confusion_matrix(y_true=Y_test2, y_pred=model_2layers_NN.predict(X_test_n))
CM_K = confusion_matrix(y_true=Y_test2, y_pred=kmeans_2.predict(X_test_Kmeans))

## predictions
test_data = pd.read_excel(r'C:\Users\10ofi\Desktop\MLproject\CarInsurance_X_test.xlsx')
features_test = test_data.drop(['ID'], axis=1)
features_test_dummy = pd.get_dummies(features_test, drop_first=False) ##convet to dummy

features_test_dummy = features_test_dummy.drop(['ANNUAL_MILEAGE','VEHICLE_TYPE_sedan','VEHICLE_TYPE_sports car'], axis=1)
X_test = minmax_scaler.transform(features_test_dummy)

test_modelNN = MLPClassifier(random_state=random_state,hidden_layer_sizes=(9, 9),
                    max_iter=261,
                    activation='relu',
                    verbose=False,
                    learning_rate_init=0.003,
                    alpha=0.00)
test_modelNN.fit(X_train_n, Y_train2)
y_pred1= test_modelNN.predict(X_test)

ans1 = pd.DataFrame();
ans1['index'] = test_data['ID']
ans1['pred'] = y_pred1
ans1.to_excel(r'C:\Users\10ofi\Desktop\MLproject\predNN.xlsx', index=False)
