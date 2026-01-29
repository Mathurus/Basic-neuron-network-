# Basic Neuron Network 


In this project, I create a simple neuron network who he just calcul the double of a number. 

For this code I use 2 librairies, numpy and matplotlib. 




# The code


For create neuron I do this : 
```
def __init__(self):
        self.poids1 = np.random.randn(1, 3) * 0.1
        self.biais1 = np.zeros((1, 3))  

        self.poids2 = np.random.randn(3, 1) * 0.1
        self.biais2 = np.zeros((1, 1))  

        print(f"Poids initiaux (petits) : {self.poids1.flatten()[:3]}")
```
This code create and connect neuron between us, and add a random weight, the weight will be change gradually of the train part. 

For the prediction : 
```
def prediction(self, X):
        self.z1 = np.dot(X, self.poids1) + self.biais1
        self.a1 = self.relu(self.z1)

        self.z2 = np.dot(self.a1, self.poids2) + self.biais2

        return self.z2
```

We add the input and the weight for do a prediction, after the script verify the prediction, because the prediction can't be under 0. So we use this code : 
```
def relu(self, x):
        return np.maximum(0, x)     

    def relu_derivee(self, x):
        return (x > 0).astype(float)
```

After we define the train part: 
```
def apprendre(self, X, y, taux_apprentissage=0.01):
        m = X.shape[0]

        prediction = self.prediction(X)

        erreur = prediction - y

        delta2 = erreur / m
        delta1 = np.dot(delta2, self.poids2.T) * self.relu_derivee(self.a1)

        self.poids2 -= taux_apprentissage * np.dot(self.a1.T, delta2)
        self.biais2 -= taux_apprentissage * np.sum(delta2, axis=0, keepdims=True)
        self.poids1 -= taux_apprentissage * np.dot(X.T, delta1)
        self.biais1 -= taux_apprentissage * np.sum(delta1, axis=0, keepdims=True)

        return np.mean(np.abs(erreur))
```

For the train part we get the first input, and we call the fonction "prediction" with the input. After we calcul the error, we subtract the prediction of the real answear who we have define. If the difine is not 0, there is an error. So we calcul do the difference between the error and the input, it's the delta. 

After this we adjust the weight and we are going to start again. 



After that, we create a loop for the train phase : 
```
reseau = ReseauMultiplication()

epochs = 5000
historique_erreurs = []

print("\n Début de l'entraînement...\n")

for epoch in range(epochs):
    erreur = reseau.apprendre(X_train, y_train, taux_apprentissage=0.01)
    historique_erreurs.append(erreur)

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} → Erreur moyenne : {erreur:.4f}")
```

The epochs is the number of train we want, if the epochs is 5000, the neuronal network are going to train 5000 times. 
And after we calcul the average of the result. 

Now lets do it with new value : 
```
for i in [0, 4, 9]:
    nombre = X_train[i][0]
    attendu = y_train[i][0]
    predit = reseau.prediction(X_train[i:i+1])[0][0]
    erreur_pct = abs(predit - attendu) / attendu * 100
    print(f"  {nombre} × 2 = {attendu} → Prédit : {predit:.2f} (erreur: {erreur_pct:.1f}%)")

print("\n Test sur des nombres JAMAIS VUS :")
nouveaux_nombres = np.array([[15], [25], [50], [100]])

for nombre in nouveaux_nombres:
    predit = reseau.prediction(nombre)[0][0]
    attendu = nombre[0] * 2
    erreur_pct = abs(predit - attendu) / attendu * 100
    print(f"  {nombre[0]} × 2 = {attendu} → Prédit : {predit:.2f} (erreur: {erreur_pct:.1f}%)")
```
So we add the new value, and calcul the prediction and the error. 

For the graphic part I used matplotlib 

# How install : 

On linux :
```
pip install -r requirements.txt
```
or 
```
pip download -r requirements.txt
```

On windows : 
```
pip install -r requirements.txt
```
Or :
```
pip download -r requirements.txt
```

After we can start the code with this command : 
```
python main.py
```

And after you can see the result with a graphic ! 

