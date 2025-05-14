# -Heart-Disease-Prediction-Using-Logistic-Regression-
Heart_Disease_Prediction.ipynb
Skip to content
Navigation Menu
talikagupta
Heart-Disease-Prediction-using-ML

Type / to search
Code
Issues
Pull requests
Actions
Projects
Security
Insights
Commit a967faa
talikagupta
talikagupta
committed
on Dec 3, 2021
Created using Colaboratory
main
1 parent 
0a0e602
 commit 
a967faa
File tree
Filter files…
Heart_Disease_Prediction.ipynb
1 file changed
+1117
-0
lines changed
Search within code
 
‎Heart_Disease_Prediction.ipynb
+1,117
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,1117 @@
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Heart Disease Prediction.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyM3WJCsFqZr4mCV/kLWeCrP",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/talikagupta/Heart-Disease-Prediction-using-ML/blob/main/Heart_Disease_Prediction.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "IcuMCo0L_gvx"
      },
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import accuracy_score"
      ],
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6LDvZdI_BsD3"
      },
      "source": [
        "Data Collection and Processing"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kLP78a3WBvAk"
      },
      "source": [
        "#loading csv file to pandas data frame\n",
        "heart_data = pd.read_csv(\"/content/heart.csv\")"
      ],
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "xBTH_kd7CukI",
        "outputId": "39718e3b-a2a8-4da5-a3ed-dd04f309e547"
      },
      "source": [
        "heart_data.head()"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>age</th>\n",
              "      <th>sex</th>\n",
              "      <th>cp</th>\n",
              "      <th>trestbps</th>\n",
              "      <th>chol</th>\n",
              "      <th>fbs</th>\n",
              "      <th>restecg</th>\n",
              "      <th>thalach</th>\n",
              "      <th>exang</th>\n",
              "      <th>oldpeak</th>\n",
              "      <th>slope</th>\n",
              "      <th>ca</th>\n",
              "      <th>thal</th>\n",
              "      <th>target</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>63</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>145</td>\n",
              "      <td>233</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>150</td>\n",
              "      <td>0</td>\n",
              "      <td>2.3</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>37</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>130</td>\n",
              "      <td>250</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>187</td>\n",
              "      <td>0</td>\n",
              "      <td>3.5</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>41</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>130</td>\n",
              "      <td>204</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>172</td>\n",
              "      <td>0</td>\n",
              "      <td>1.4</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>56</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>120</td>\n",
              "      <td>236</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>178</td>\n",
              "      <td>0</td>\n",
              "      <td>0.8</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>57</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>120</td>\n",
              "      <td>354</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>163</td>\n",
              "      <td>1</td>\n",
              "      <td>0.6</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   age  sex  cp  trestbps  chol  fbs  ...  exang  oldpeak  slope  ca  thal  target\n",
              "0   63    1   3       145   233    1  ...      0      2.3      0   0     1       1\n",
              "1   37    1   2       130   250    0  ...      0      3.5      0   0     2       1\n",
              "2   41    0   1       130   204    0  ...      0      1.4      2   0     2       1\n",
              "3   56    1   1       120   236    0  ...      0      0.8      2   0     2       1\n",
              "4   57    0   0       120   354    0  ...      1      0.6      2   0     2       1\n",
              "\n",
              "[5 rows x 14 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "eWBJPI68C4is",
        "outputId": "9228d983-5fff-4b45-f43f-0bf73199f50a"
      },
      "source": [
        "heart_data.tail()"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>age</th>\n",
              "      <th>sex</th>\n",
              "      <th>cp</th>\n",
              "      <th>trestbps</th>\n",
              "      <th>chol</th>\n",
              "      <th>fbs</th>\n",
              "      <th>restecg</th>\n",
              "      <th>thalach</th>\n",
              "      <th>exang</th>\n",
              "      <th>oldpeak</th>\n",
              "      <th>slope</th>\n",
              "      <th>ca</th>\n",
              "      <th>thal</th>\n",
              "      <th>target</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>298</th>\n",
              "      <td>57</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>140</td>\n",
              "      <td>241</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>123</td>\n",
              "      <td>1</td>\n",
              "      <td>0.2</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>299</th>\n",
              "      <td>45</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>110</td>\n",
              "      <td>264</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>132</td>\n",
              "      <td>0</td>\n",
              "      <td>1.2</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>300</th>\n",
              "      <td>68</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>144</td>\n",
              "      <td>193</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>141</td>\n",
              "      <td>0</td>\n",
              "      <td>3.4</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>301</th>\n",
              "      <td>57</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>130</td>\n",
              "      <td>131</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>115</td>\n",
              "      <td>1</td>\n",
              "      <td>1.2</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>302</th>\n",
              "      <td>57</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>130</td>\n",
              "      <td>236</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>174</td>\n",
              "      <td>0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "     age  sex  cp  trestbps  chol  fbs  ...  exang  oldpeak  slope  ca  thal  target\n",
              "298   57    0   0       140   241    0  ...      1      0.2      1   0     3       0\n",
              "299   45    1   3       110   264    0  ...      0      1.2      1   0     3       0\n",
              "300   68    1   0       144   193    1  ...      0      3.4      1   2     3       0\n",
              "301   57    1   0       130   131    0  ...      1      1.2      1   1     3       0\n",
              "302   57    0   1       130   236    0  ...      0      0.0      1   1     2       0\n",
              "\n",
              "[5 rows x 14 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0-LuqjB9DH7u",
        "outputId": "03fe09d4-bb4a-46d5-fc3f-9e8bd35bc53b"
      },
      "source": [
        "heart_data.shape  #303 rows , 14 columns"
      ],
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(303, 14)"
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-6jNI1VtDi4H",
        "outputId": "6c92065d-3aeb-4ca8-bf39-1527bdd58a53"
      },
      "source": [
        "heart_data.info()"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 303 entries, 0 to 302\n",
            "Data columns (total 14 columns):\n",
            " #   Column    Non-Null Count  Dtype  \n",
            "---  ------    --------------  -----  \n",
            " 0   age       303 non-null    int64  \n",
            " 1   sex       303 non-null    int64  \n",
            " 2   cp        303 non-null    int64  \n",
            " 3   trestbps  303 non-null    int64  \n",
            " 4   chol      303 non-null    int64  \n",
            " 5   fbs       303 non-null    int64  \n",
            " 6   restecg   303 non-null    int64  \n",
            " 7   thalach   303 non-null    int64  \n",
            " 8   exang     303 non-null    int64  \n",
            " 9   oldpeak   303 non-null    float64\n",
            " 10  slope     303 non-null    int64  \n",
            " 11  ca        303 non-null    int64  \n",
            " 12  thal      303 non-null    int64  \n",
            " 13  target    303 non-null    int64  \n",
            "dtypes: float64(1), int64(13)\n",
            "memory usage: 33.3 KB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "odQWb_DCFATs",
        "outputId": "ad2b3702-7237-4dbb-b882-3d52860020e3"
      },
      "source": [
        "#check for missing values\n",
        "heart_data.isnull().sum()"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "age         0\n",
              "sex         0\n",
              "cp          0\n",
              "trestbps    0\n",
              "chol        0\n",
              "fbs         0\n",
              "restecg     0\n",
              "thalach     0\n",
              "exang       0\n",
              "oldpeak     0\n",
              "slope       0\n",
              "ca          0\n",
              "thal        0\n",
              "target      0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 320
        },
        "id": "-kJqYU2uF4RX",
        "outputId": "d125f2ae-6bc2-44fa-c53d-27d9261ddc13"
      },
      "source": [
        "#statistical measures of the data\n",
        "heart_data.describe()"
      ],
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>age</th>\n",
              "      <th>sex</th>\n",
              "      <th>cp</th>\n",
              "      <th>trestbps</th>\n",
              "      <th>chol</th>\n",
              "      <th>fbs</th>\n",
              "      <th>restecg</th>\n",
              "      <th>thalach</th>\n",
              "      <th>exang</th>\n",
              "      <th>oldpeak</th>\n",
              "      <th>slope</th>\n",
              "      <th>ca</th>\n",
              "      <th>thal</th>\n",
              "      <th>target</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>count</th>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "      <td>303.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>mean</th>\n",
              "      <td>54.366337</td>\n",
              "      <td>0.683168</td>\n",
              "      <td>0.966997</td>\n",
              "      <td>131.623762</td>\n",
              "      <td>246.264026</td>\n",
              "      <td>0.148515</td>\n",
              "      <td>0.528053</td>\n",
              "      <td>149.646865</td>\n",
              "      <td>0.326733</td>\n",
              "      <td>1.039604</td>\n",
              "      <td>1.399340</td>\n",
              "      <td>0.729373</td>\n",
              "      <td>2.313531</td>\n",
              "      <td>0.544554</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>std</th>\n",
              "      <td>9.082101</td>\n",
              "      <td>0.466011</td>\n",
              "      <td>1.032052</td>\n",
              "      <td>17.538143</td>\n",
              "      <td>51.830751</td>\n",
              "      <td>0.356198</td>\n",
              "      <td>0.525860</td>\n",
              "      <td>22.905161</td>\n",
              "      <td>0.469794</td>\n",
              "      <td>1.161075</td>\n",
              "      <td>0.616226</td>\n",
              "      <td>1.022606</td>\n",
              "      <td>0.612277</td>\n",
              "      <td>0.498835</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>min</th>\n",
              "      <td>29.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>94.000000</td>\n",
              "      <td>126.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>71.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>25%</th>\n",
              "      <td>47.500000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>120.000000</td>\n",
              "      <td>211.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>133.500000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>0.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>50%</th>\n",
              "      <td>55.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>130.000000</td>\n",
              "      <td>240.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>153.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>0.800000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>75%</th>\n",
              "      <td>61.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>140.000000</td>\n",
              "      <td>274.500000</td>\n",
              "      <td>0.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>166.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>1.600000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>max</th>\n",
              "      <td>77.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>200.000000</td>\n",
              "      <td>564.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>202.000000</td>\n",
              "      <td>1.000000</td>\n",
              "      <td>6.200000</td>\n",
              "      <td>2.000000</td>\n",
              "      <td>4.000000</td>\n",
              "      <td>3.000000</td>\n",
              "      <td>1.000000</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "              age         sex          cp  ...          ca        thal      target\n",
              "count  303.000000  303.000000  303.000000  ...  303.000000  303.000000  303.000000\n",
              "mean    54.366337    0.683168    0.966997  ...    0.729373    2.313531    0.544554\n",
              "std      9.082101    0.466011    1.032052  ...    1.022606    0.612277    0.498835\n",
              "min     29.000000    0.000000    0.000000  ...    0.000000    0.000000    0.000000\n",
              "25%     47.500000    0.000000    0.000000  ...    0.000000    2.000000    0.000000\n",
              "50%     55.000000    1.000000    1.000000  ...    0.000000    2.000000    1.000000\n",
              "75%     61.000000    1.000000    2.000000  ...    1.000000    3.000000    1.000000\n",
              "max     77.000000    1.000000    3.000000  ...    4.000000    3.000000    1.000000\n",
              "\n",
              "[8 rows x 14 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uIC0JNJ0GHiO",
        "outputId": "37ac2ece-5fad-4fdb-c5e2-0d945ac9ccde"
      },
      "source": [
        "#checking the distribution of the target variable for checking if it is a balanced data set or a skewed one\n",
        "heart_data['target'].value_counts()  #1-defective heart 0-healthy heart"
      ],
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1    165\n",
              "0    138\n",
              "Name: target, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "1TGsym2QHri-"
      },
      "source": [
        "Splitting the feature columns and the target column"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8HDJp29IHnaB"
      },
      "source": [
        "X = heart_data.drop(columns='target', axis = 1)\n",
        "Y = heart_data['target']"
      ],
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ywsyylCRMyyZ",
        "outputId": "c41232c5-e717-4cfe-b816-b396337c122f"
      },
      "source": [
        "print(X)"
      ],
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "     age  sex  cp  trestbps  chol  ...  exang  oldpeak  slope  ca  thal\n",
            "0     63    1   3       145   233  ...      0      2.3      0   0     1\n",
            "1     37    1   2       130   250  ...      0      3.5      0   0     2\n",
            "2     41    0   1       130   204  ...      0      1.4      2   0     2\n",
            "3     56    1   1       120   236  ...      0      0.8      2   0     2\n",
            "4     57    0   0       120   354  ...      1      0.6      2   0     2\n",
            "..   ...  ...  ..       ...   ...  ...    ...      ...    ...  ..   ...\n",
            "298   57    0   0       140   241  ...      1      0.2      1   0     3\n",
            "299   45    1   3       110   264  ...      0      1.2      1   0     3\n",
            "300   68    1   0       144   193  ...      0      3.4      1   2     3\n",
            "301   57    1   0       130   131  ...      1      1.2      1   1     3\n",
            "302   57    0   1       130   236  ...      0      0.0      1   1     2\n",
            "\n",
            "[303 rows x 13 columns]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "M2d7pglyNJma",
        "outputId": "fda2ac64-64f5-4cdc-e737-690116d1eb22"
      },
      "source": [
        "print(Y)"
      ],
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "0      1\n",
            "1      1\n",
            "2      1\n",
            "3      1\n",
            "4      1\n",
            "      ..\n",
            "298    0\n",
            "299    0\n",
            "300    0\n",
            "301    0\n",
            "302    0\n",
            "Name: target, Length: 303, dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "enijNhpMO4zB"
      },
      "source": [
        "X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)"
      ],
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "r5ZmpB4lNLmt",
        "outputId": "560ab95e-b47a-477f-db09-4278e05d9f0e"
      },
      "source": [
        "print(X.shape, X_train.shape, X_test.shape)"
      ],
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(303, 13) (242, 13) (61, 13)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sSHxPK7XPzrl",
        "outputId": "ff0f506e-939b-40f4-b722-e3feba0998de"
      },
      "source": [
        "print(X_train, Y_train)"
      ],
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "     age  sex  cp  trestbps  chol  ...  exang  oldpeak  slope  ca  thal\n",
            "19    69    0   3       140   239  ...      0      1.8      2   2     2\n",
            "247   66    1   1       160   246  ...      1      0.0      1   3     1\n",
            "289   55    0   0       128   205  ...      1      2.0      1   1     3\n",
            "288   57    1   0       110   335  ...      1      3.0      1   1     3\n",
            "60    71    0   2       110   265  ...      0      0.0      2   1     2\n",
            "..   ...  ...  ..       ...   ...  ...    ...      ...    ...  ..   ...\n",
            "39    65    0   2       160   360  ...      0      0.8      2   0     2\n",
            "104   50    1   2       129   196  ...      0      0.0      2   0     2\n",
            "140   51    0   2       120   295  ...      0      0.6      2   0     2\n",
            "114   55    1   1       130   262  ...      0      0.0      2   0     2\n",
            "110   64    0   0       180   325  ...      1      0.0      2   0     2\n",
            "\n",
            "[242 rows x 13 columns] 19     1\n",
            "247    0\n",
            "289    0\n",
            "288    0\n",
            "60     1\n",
            "      ..\n",
            "39     1\n",
            "104    1\n",
            "140    1\n",
            "114    1\n",
            "110    1\n",
            "Name: target, Length: 242, dtype: int64\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "t_PPJ6gbYtRG"
      },
      "source": [
        "Model Training"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "1lSvmJpDYxah"
      },
      "source": [
        "Logistic Regression"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wo4K1Iz2Y1oT"
      },
      "source": [
        "model = LogisticRegression()"
      ],
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TalXk-q3Y0St",
        "outputId": "4f45012a-585e-46ed-da21-e5bf3c2376d4"
      },
      "source": [
        "# training the LogisticRegression model with Training data\n",
        "model.fit(X_train, Y_train)"
      ],
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_logistic.py:818: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
            "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
            "\n",
            "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
            "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
            "Please also refer to the documentation for alternative solver options:\n",
            "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
            "  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LogisticRegression()"
            ]
          },
          "metadata": {},
          "execution_count": 17
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "jLzZHcadZTwK"
      },
      "source": [
        "Model Evaluation\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "YX6p0Cd0ZTkH"
      },
      "source": [
        "Accuracy Score\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rRsHsc0uZXDd",
        "outputId": "347de234-1cef-420f-8f51-64d145cf71f0"
      },
      "source": [
        "#accuracy on training data\n",
        "X_train_prediction = model.predict(X_train)\n",
        "training_data_accuracy = accuracy_score(X_train_prediction, Y_train)\n",
        "print('Accuracy on Training data : ', training_data_accuracy)"
      ],
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy on Training data :  0.8471074380165289\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YWGL822vcVNB",
        "outputId": "dc9cc608-b9ef-4411-c87e-fc7c5cf8c2dd"
      },
      "source": [
        "#accuracy on test data\n",
        "X_test_prediction = model.predict(X_test)\n",
        "test_data_accuracy = accuracy_score(X_test_prediction, Y_test)\n",
        "print('Accuracy on Test data : ', test_data_accuracy)"
      ],
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy on Test data :  0.8032786885245902\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_176iHTCcud1"
      },
      "source": [
        "#if accuracy test << accuracy training, then we have an overfitting problem"
      ],
      "execution_count": 20,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "P53NvFvac75X"
      },
      "source": [
        "Building a predictive system\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vNk8lLQuc7n0",
        "outputId": "6ba52e57-b905-4d50-dc5d-f56cd7ac8f6a"
      },
      "source": [
        "input_data = (37,1,3,145,233,1,0,150,0,2.3,0,0,2)\n",
        "\n",
        "# change the input data to a numpy array\n",
        "input_data_as_numpy_array= np.asarray(input_data)\n",
        "\n",
        "# reshape the numpy array as we are predicting for only on instance\n",
        "input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)\n",
        "#column dim is unknown(we want numpy to figure it out), row = 1\n",
        "\n",
        "prediction = model.predict(input_data_reshaped)\n",
        "print(prediction)\n",
        "\n",
        "if (prediction[0]== 0):\n",
        "  print('The Person does not have a Heart Disease')\n",
        "else:\n",
        "  print('The Person has Heart Disease')"
      ],
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[1]\n",
            "The Person has Heart Disease\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/sklearn/base.py:446: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names\n",
            "  \"X does not have valid feature names, but\"\n"
          ]
        }
      ]
    }
  ]
}
0 commit comments
Comments
0
 (0)
Comment
You're not receiving notifications from this thread.

Copied!  
