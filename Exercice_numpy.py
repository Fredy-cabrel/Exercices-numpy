# Exercice sur NumPy
# Exo 1
# Chaque ligne représente un étudiant et chaque colonne un examen.

#Questions
#1- Afficher la forme du tableau.
#2- Afficher le nombre de dimensions.
#3- Calculer la moyenne de toutes les notes.
#4- Calculer la moyenne de chaque étudiant.
#5- Calculer la moyenne de chaque examen.
#6- Trouver la note maximale.
#7- Trouver la note minimale.
#8- Afficher les notes du 3e étudiant.
#9- Afficher les notes du 2e examen.

import numpy as np

notes = np.array([
    [70, 80, 90],
    [60, 75, 85],
    [90, 95, 100],
    [50, 65, 70]
])

#1- Afficher la forme du tableau.
forme = notes.shape
print("Forme du tableau:", forme)

#2- Afficher le nombre de dimensions.
dimensions = notes.ndim
print("Nombre de dimensions:", dimensions)

#3- Calculer la moyenne de toutes les notes.
moyenne = np.mean(notes)
print("Moyenne de toutes les notes:", moyenne)

#4- Calculer la moyenne de chaque étudiant.
moy_etudiant = np.mean(notes, axis = 1)
print("Moyenne de chaque étudiant:", moy_etudiant)

#5- Calculer la moyenne de chaque examen.
moy_examen = np.mean(notes, axis = 0)
print("Moyenne de chaque examen:", moy_examen)

#6- Trouver la note maximale.
note_maximale = np.max(notes)
print("Note maximale:", note_maximale)

#7- Trouver la note minimale.
note_minimale = np.min(notes)
print("Note minimale:", note_minimale)

#8- Afficher les notes du 3e étudiant.
notes_3e_etudiant = notes[2]
print("Notes du 3e étudiant:", notes_3e_etudiant)

#9- Afficher les notes du 2e examen.
notes_2e_examen = notes[:, 1]
print("Notes du 2e examen:", notes_2e_examen)

# Exo 2
tab = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

#Sans exécuter le code, essaye de deviner :
print("Exercice 2 :")
print(np.sum(tab))
print(np.sum(tab, axis=0))
print(np.sum(tab, axis=1))

# Exo 3

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print("Exercice 3 :")
print("la concaténation de a et b est : ", np.concatenate((a, b), axis = 0)) # concaténation le long de l'axe 0 (lignes)
print("la concaténation de a et b est : ", np.concatenate((a, b), axis = 1)) # concaténation le long de l'axe 1 (colonnes)

print("l'empilement vertical est : ", np.vstack((a, b))) # empilement vertical de a et b
print("l'empilement horizontal est : ", np.hstack((a, b))) # empilement horizontal de a et b


# Exo 4
tab = np.array([1, 2, 3, 4, 5, 6])

print("Exercice 4 :")
print("le tableau numpy est : ", tab)
print("Le tableau reshape est : ", np.reshape(tab, (2, 3))) # reshape le tableau en une matrice de 2 lignes et 3 colonnes
print("la dimension du tableau reshape est : ", np.reshape(tab, (2, 3)).ndim) # affiche le nombre de dimensions du tableau reshape
print("Le tableau reshape est : ", np.reshape(tab, (3, 2))) # reshape le tableau en une matrice de 3 lignes et 2 colonnes
print("la dimension du tableau reshape est : ", np.reshape(tab, (3, 2)).ndim)
print("Le tableau reshape est : ", np.reshape(tab, (1, 6))) # reshape le tableau en une matrice de 1 ligne et 6 colonnes
print("la dimension du tableau reshape est : ", np.reshape(tab, (1, 6)).ndim)
print("Le tableau reshape est : ", np.reshape(tab, (6, 1))) # reshape le tableau en une matrice de 1 ligne et 6 colonnes
print("la dimension du tableau reshape est : ", np.reshape(tab, (6, 1)).ndim)

# Exo 5 : Mini perceptron
# Sans créer de classe, fais simplement :

print("Exercice 5 :")
x = np.array([1, 0])
poids = np.array([0.4, 0.6])
biais = -0.2

# calculons :
a = np.dot(x, poids)

# calculons :
b = np.dot(x, poids) + biais

print(a.shape, b.shape)

if b >= 0 :
    print(1)
else :
    print(0)



# Exo 5 : niveau entretien/stagevente_std_magasin

print("Exercice 5 : ")

ventes = np.array([
    [100, 150, 200],
    [120, 180, 220],
    [90, 140, 210]
])

# Lignes = magasins
# Colonnes = mois

# 1. Les ventes totales de chaque magasin.
ventes_totales_mag = np.sum(ventes, axis = 1)
print(" Les ventes totales de chaque magasin sont : ", ventes_totales_mag)

# 2. Les ventes totales de chaque mois.
ventes_totales_mois = np.sum(ventes, axis = 0)
print(" Les ventes totales de chaque mois sont : ", ventes_totales_mois)

# 3. Le magasin qui a vendu le plus.
vente_meilleur_magasin = np.argmax(ventes_totales_mag)
print(" Le magasin qui a vendu le plus est : ", vente_meilleur_magasin)

# 4. Le mois qui a généré le plus de ventes.
vente_meilleur_mois = np.argmax(ventes_totales_mois)
print(" Le mois qui a généré le plus de ventes est : ", vente_meilleur_mois)

#5. L'écart-type des ventes par magasin.
vente_std_magasin = np.std(ventes, axis = 1)
print(" L'écart-type des ventes par magasin est : ", vente_std_magasin)

#5. L'écart-type des ventes par mois.
vente_std_mois = np.std(ventes, axis = 0)
print(" L'écart-type des ventes par mois est : ", vente_std_mois)


