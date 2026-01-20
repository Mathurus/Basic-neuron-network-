import numpy as np 
import matplotlib.pyplot as plt

X_train = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y_train = np.array([[2], [4], [6], [8], [10], [12], [14], [16], [18], [20]])

print("=" * 60)
print("Exemples d'entraînement :")
for i in range(5):
    print(f"  {X_train[i][0]} × 2 = {y_train[i][0]}")
print("=" * 60)

class ReseauMultiplication:
    def __init__(self):
        self.poids1 = np.random.randn(1, 3) * 0.1
        self.biais1 = np.zeros((1, 3))  

        self.poids2 = np.random.randn(3, 1) * 0.1
        self.biais2 = np.zeros((1, 1))  

        print(f"Poids initiaux (petits) : {self.poids1.flatten()[:3]}")

    def relu(self, x):
        return np.maximum(0, x)     

    def relu_derivee(self, x):
        return (x > 0).astype(float)

    def prediction(self, X):
        self.z1 = np.dot(X, self.poids1) + self.biais1
        self.a1 = self.relu(self.z1)

        self.z2 = np.dot(self.a1, self.poids2) + self.biais2

        return self.z2

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

reseau = ReseauMultiplication()

epochs = 5000
historique_erreurs = []

print("\n Début de l'entraînement...\n")

for epoch in range(epochs):
    erreur = reseau.apprendre(X_train, y_train, taux_apprentissage=0.01)
    historique_erreurs.append(erreur)

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} → Erreur moyenne : {erreur:.4f}")

print("\n Entraînement terminé !")

print("\n" + "=" * 60)
print("Test sur les données connues :")
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

print("=" * 60)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(historique_erreurs, color='blue', linewidth=2)
plt.title("Apprentissage : Réduction de l'erreur", fontsize=14, fontweight='bold')
plt.xlabel("Itération (Epoch)")
plt.ylabel("Erreur moyenne")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
nombres_test = np.array([[i] for i in range(0, 21)])
predictions = reseau.prediction(nombres_test)

plt.plot(nombres_test, nombres_test * 2, 'g-', linewidth=2, label='Réalité (× 2)')
plt.plot(nombres_test, predictions, 'r--', linewidth=2, label='Prédictions du réseau')
plt.scatter(X_train, y_train, color='blue', s=50, label='Données d\'entraînement', zorder=5)
plt.title("Prédictions du réseau", fontsize=14, fontweight='bold')
plt.xlabel("Nombre d'entrée")
plt.ylabel("Résultat")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
