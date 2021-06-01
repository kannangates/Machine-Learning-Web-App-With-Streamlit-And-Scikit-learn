import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import streamlit.components.v1 as stc
import numpy as np
from sklearn import datasets


HTML_BANNER = """
    <div style="background-color:#446e5f;padding:5px;border-radius:10px">
    <h1 style="color:white;text-align:center;">Streamlit app for Machine Learning</h1>
    <h3 style="color:white;text-align:center;">Check which classifier can run the best</h3>
    </div>
    """
stc.html(HTML_BANNER)


#st.title("Streamlit app for Machine Learning")
#st.text("Check which classifier can run the best")

dataset_name = st.selectbox("Select a Dataset",
                            ("Iris", "Breast Cancer", "Wine Dataset"))
classifier_name = st.selectbox("Select which classifier you want to check with",
                               ("KNN", "SVM", "Random Forest"))

# st.title(f"You have selected the dataset : {dataset_name} and classifier {classifier_name}")


def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y


X, y = get_dataset(dataset_name)
len_y = len(np.unique(y))
st.write("The Shape of the data:", X.shape)
st.write(
    f"Number of labels/Class in the target data is {len_y} and they are as follows", (np.unique(y)))


def get_param(clf_name):
    params = dict()
    if clf_name == "KNN":
        st.write("Select K, Nearest Neighbours")
        K = st.slider("K Numbers", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        st.write("Select C, Regulalrization Parameter")
        C = st.slider("C Numbers", 0.01, 10.0)
        params["C"] = C
    else:
        criterion = st.radio("Select criterion", ("gini", "entropy"))
        max_depth = st.slider("Maximum Depth", 2, 15)
        n_estimators = st.slider("n_estimators", 1, 100)
        random_state = st.slider("random_state", 1, 100)
        params["criterion"] = criterion
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        params["random_state"] = random_state
    return params


params = get_param(classifier_name)


def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(criterion=params["criterion"], max_depth=params["max_depth"],
                                     n_estimators=params["n_estimators"], random_state=params["random_state"])
    return clf


st.subheader("Train_Test_Split data")
te_size = st.slider(
    "test_Size", min_value=0.2, max_value=0.35)
test_random_state = st.slider("Test random_state", 1, 100)

clf = get_classifier(classifier_name, params)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, random_state=test_random_state, test_size=te_size)
model = clf.fit(Xtrain, ytrain)
ypredict = model.predict(Xtest)

acc_score = accuracy_score(ytest, ypredict)
st.subheader(f"Selected Classifier: {classifier_name} Classifier")
st.subheader(f"Accuracy_Score for this data is {acc_score}")

# PLOTS
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Componets 1")
plt.ylabel("Principal Componets 2")
plt.colorbar()

st.pyplot(fig)
st.write("Thank You")
