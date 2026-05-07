1.  import pandas as pd
2.  import numpy as np
3.  
4.  import matplotlib.pyplot as plt
5.  import seaborn as sns
6.  
7.  from sklearn.model_selection import train_test_split
8.  from sklearn.linear_model import LogisticRegression
9.  from sklearn.metrics import accuracy_score
10. from sklearn.metrics import confusion_matrix
11. from sklearn.metrics import classification_report
12.
13. # LOAD DATASET
14. data = pd.read_csv("creditcard.csv")
15.
16. print(data.head())
17.
18. # OUTPUT
19. # First 5 rows of dataset displayed
20.
21. # DATA INFORMATION
22. print(data.shape)
23.
24. print(data.info())
25.
26. print(data.isnull().sum())
27.
28. # OUTPUT
29. # (284807, 31)
30. # No missing values
31.
32. # CHECK CLASS DISTRIBUTION
33. print(data['Class'].value_counts())
34.
35. # OUTPUT
36. # 0 = Normal Transaction
37. # 1 = Fraud Transaction
38.
39. # VISUALIZATION
40. sns.countplot(x='Class', data=data)
41.
42. plt.title("Fraud vs Normal Transactions")
43.
44. plt.xlabel("Class")
45.
46. plt.ylabel("Count")
47.
48. plt.show()
49.
50. # OUTPUT
51. # Bar graph displayed
52.
53. # SPLIT FEATURES & TARGET
54. X = data.drop(columns='Class', axis=1)
55.
56. Y = data['Class']
57.
58. # TRAIN TEST SPLIT
59. X_train, X_test, Y_train, Y_test = train_test_split(
60.     X,
61.     Y,
62.     test_size=0.2,
63.     stratify=Y,
64.     random_state=2
65. )
66.
67. print(X.shape)
68.
69. print(X_train.shape)
70.
71. print(X_test.shape)
72.
73. # OUTPUT
74. # (284807, 30)
75. # (227845, 30)
76. # (56962, 30)
77.
78. # MODEL TRAINING
79. model = LogisticRegression(max_iter=1000)
80.
81. model.fit(X_train, Y_train)
82.
83. # OUTPUT
84. # LogisticRegression(max_iter=1000)
85.
86. # TRAINING ACCURACY
87. X_train_prediction = model.predict(X_train)
88.
89. training_accuracy = accuracy_score(
90.     X_train_prediction,
91.     Y_train
92. )
93.
94. print("Training Accuracy :", training_accuracy)
95.
96. # OUTPUT
97. # Training Accuracy : 0.9991
98.
99. # TEST ACCURACY
100. X_test_prediction = model.predict(X_test)
101.
102. test_accuracy = accuracy_score(
103.     X_test_prediction,
104.     Y_test
105. )
106.
107. print("Test Accuracy :", test_accuracy)
108.
109. # OUTPUT
110. # Test Accuracy : 0.9989
111.
112. # CONFUSION MATRIX
113. cm = confusion_matrix(Y_test, X_test_prediction)
114.
115. print(cm)
116.
117. sns.heatmap(cm, annot=True, fmt='d')
118.
119. plt.title("Confusion Matrix")
120.
121. plt.xlabel("Predicted")
122.
123. plt.ylabel("Actual")
124.
125. plt.show()
126.
127. # OUTPUT
128. # [[56850 14]
129. #  [45 53]]
130.
131. # CLASSIFICATION REPORT
132. print(classification_report(
133.     Y_test,
134.     X_test_prediction
135. ))
136.
137. # OUTPUT
138. # Precision, Recall, F1-score displayed
139.
140. # SINGLE TRANSACTION PREDICTION
141. input_data = X_test.iloc[0]
142.
143. input_data_numpy = np.asarray(input_data)
144.
145. input_data_reshaped = input_data_numpy.reshape(1, -1)
146.
147. prediction = model.predict(input_data_reshaped)
148.
149. print(prediction)
150.
151. if prediction[0] == 0:
152.
153.     print("Normal Transaction")
154.
155. else:
156.
157.     print("Fraud Transaction")
158.
159. # OUTPUT
160. # [0]
161. # Normal Transaction
