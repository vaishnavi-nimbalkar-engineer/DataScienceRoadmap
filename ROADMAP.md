# Datenanalyse und Data Science mit Python: Eine Roadmap

## Inhaltsverzeichnis

1.  **Erste Schritte und Umgebungseinrichtung**
    *   1.1 Python installieren und Entwicklungsumgebung einrichten
    *   1.2 Einführung in Jupyter Notebooks
    *   1.3 Virtuelle Umgebungen nutzen
    *   1.4 Code und Daten herunterladen
2.  **Grundlagen von Python für die Datenanalyse**
    *   2.1 Python Grundlagen
    *   2.2 Datentypen und Datenstrukturen
    *   2.3 Funktionen und Module
    *   2.4 Dateiverarbeitung
3.  **Datenmanipulation mit Pandas**
    *   3.1 Einführung in Pandas DataFrames und Series
    *   3.2 Daten laden, speichern und Dateiformate
    *   3.3 Datenbereinigung und -vorbereitung (Data Wrangling)
    *   3.4 Datentransformation
    *   3.5 Datenmerging und -abfragen
    *   3.6 Umgang mit Zeitreihendaten
4.  **Datenvisualisierung mit Matplotlib und Seaborn**
    *   4.1 Einführung in Matplotlib
    *   4.2 Erstellen verschiedener Diagrammtypen (Linien-, Balken-, Streudiagramme etc.)
    *   4.3 Anpassung und Formatierung von Plots
    *   4.4 Fortgeschrittene Visualisierungen mit Seaborn
5.  **Explorative Datenanalyse (EDA) und Statistik**
    *   5.1 Grundlagen der explorativen Datenanalyse
    *   5.2 Deskriptive Statistik
    *   5.3 Korrelation
    *   5.4 Hypothesentest und Regression
6.  **Einführung in Machine Learning**
    *   6.1 Überblick über Machine Learning
    *   6.2 Datenvorverarbeitung für Machine Learning
    *   6.3 Beispiele für Machine Learning Modelle

---

## 1. Erste Schritte und Umgebungseinrichtung

*   **1.1 Python installieren und Entwicklungsumgebung einrichten:**
    *   **Beschreibung:** Dieser Abschnitt behandelt die notwendigen Schritte zur Installation von Python und zur Einrichtung einer geeigneten Entwicklungsumgebung für die Datenanalyse. Dazu gehört die Auswahl und Installation einer Python-Distribution wie **Anaconda**, die viele nützliche Pakete für Data Science bereits enthält. Alternativ kann Python auch über `pip` installiert werden. Die Verwendung von integrierten Entwicklungsumgebungen (IDEs) oder Texteditoren wird ebenfalls angesprochen. Das **Enthought Canopy** wird als eine Distribution erwähnt, bei der die meisten für dieses Buch benötigten Pakete vorinstalliert sind.
    *   **Codebeispiel (Installation mit Anaconda):**
        ```bash
        # Beispielhafter Befehl zum Installieren von Pandas mit conda
        conda install pandas
        ```
    *   **Codebeispiel (Installation mit pip):**
        ```bash
        # Beispielhafter Befehl zum Installieren von Pandas mit pip
        pip install pandas
        ```

*   **1.2 Einführung in Jupyter Notebooks:**
    *   **Beschreibung:** **Jupyter Notebooks** sind eine interaktive Umgebung, die es ermöglicht, Code, Text (in Markdown und HTML), mathematische Formeln, Visualisierungen und andere Multimedia-Inhalte in einem einzigen Dokument zu kombinieren [1.2, 88, 102, 131]. Sie sind besonders nützlich für die explorative Datenanalyse, da Codeblöcke einzeln ausgeführt und die Ergebnisse direkt darunter angezeigt werden können. Jupyter Notebooks werden in der Data-Science-Welt häufig verwendet. Die Dateien haben die Erweiterung `.ipynb`. Das **Magic Command `%matplotlib inline`** ermöglicht die Anzeige von Matplotlib-Plots direkt im Notebook.
    *   **Codebeispiel (Ausführen einer Zelle in Jupyter Notebook):**
        ```python
        # Dies ist eine Codezelle
        import pandas as pd
        print("Pandas wurde importiert!")
        ```
        Um diese Zelle auszuführen, drückt man **Shift + Enter**.
    *   **Codebeispiel (Markdown-Zelle):**
        ```markdown
        # Dies ist eine Markdown-Zelle
        Hier können Sie Text, Listen, Bilder und mehr einfügen.
        ```
        Diese Zelle wird als formatierter Text angezeigt, nachdem sie ausgeführt wurde.

*   **1.3 Virtuelle Umgebungen nutzen:**
    *   **Beschreibung:** **Virtuelle Umgebungen** dienen dazu, Projektabhängigkeiten zu isolieren [1.3, 4, 17]. Dies verhindert Konflikte zwischen verschiedenen Projekten, die möglicherweise unterschiedliche Versionen derselben Bibliotheken benötigen. Tools wie das `venv`-Modul von Python und **conda** können zur Erstellung und Verwaltung solcher Umgebungen verwendet werden. Vor der Installation von Paketen mit `pip` sollte die entsprechende virtuelle Umgebung aktiviert werden [1.3, 19].
    *   **Codebeispiel (Erstellung einer virtuellen Umgebung mit venv unter Windows):**
        ```bash
        C:\...> python3 -m venv book_env
        C:\...> book_env\Scripts\activate
        (book_env) C:\...>
        ```
    *   **Codebeispiel (Erstellung einer conda-Umgebung):**
        ```bash
        conda create -n my_data_env python=3.8
        conda activate my_data_env
        ```

*   **1.4 Code und Daten herunterladen:**
    *   **Beschreibung:** Begleitender Code und Datensätze für Bücher im Bereich Datenanalyse sind oft auf **GitHub** verfügbar [1.4, 6, 8, 12, 24, 25, 31, 39, 41, 42, 63, 68, 70, 72, 74, 75, 76, 78, 128, 133, 161]. Die Quellen geben spezifische URLs zu diesen Repositories an. Es wird empfohlen, den Code selbst abzutippen, um Fehler zu vermeiden [1.4, 62]. Der Befehl `git clone` wird verwendet, um ein Repository von GitHub herunterzuladen [1.4, 32]. Es wird auch die Möglichkeit erwähnt, Repositories zu **forken** und dann zu klonen, um eigene Änderungen zu verfolgen und hochzuladen.
    *   **Codebeispiel (Klonen eines GitHub-Repository):**
        ```bash
        git clone https://github.com/stefmolin/Hands-On-Data-Analysis-with-Pandas-2nd-edition.git
        ```

## 2. Grundlagen von Python für die Datenanalyse

*   **2.1 Python Grundlagen:**
    *   **Beschreibung:** Dieser Abschnitt setzt grundlegende Kenntnisse der **Python-Syntax** voraus [2.1]. Dazu gehören Konzepte wie Variablen, Operatoren, Kontrollfluss (z.B. `if`-Anweisungen, Schleifen), Module, Listen und Tupel [2.1, 56]. Die Betonung liegt auf der Lesbarkeit und Einfachheit der Python-Sprache. Kommentare werden mit `#` eingeleitet und vom Interpreter ignoriert.
    *   **Codebeispiel (Grundlegende Python-Syntax):**
        ```python
        # Dies ist ein Kommentar
        x = 5
        y = 10
        if x < y:
            print("x ist kleiner als y")
        for i in range(5):
            print(i)
        meine_liste =
        mein_tupel = (4, 5, 6)
        ```

*   **2.2 Datentypen und Datenstrukturen:**
    *   **Beschreibung:** Python bietet verschiedene integrierte **Datentypen** wie Integer, Float, String und Boolean sowie **Datenstrukturen** wie Listen, Tupel und Dictionaries [2.2]. Listen sind veränderliche Sequenzen, Tupel sind unveränderliche Sequenzen und Dictionaries speichern Schlüssel-Wert-Paare [2.2]. Das Verständnis dieser Strukturen ist für die Datenmanipulation unerlässlich [2.2, 90].
    *   **Codebeispiel (Datentypen und Datenstrukturen):**
        ```python
        zahl = 10             # Integer
        kommazahl = 3.14      # Float
        text = "Hallo"        # String
        wahrheit = True       # Boolean
        meine_liste = [1, "zwei", 3.0]
        mein_tupel = (1, 2, 3)
        mein_dict = {"name": "Max", "alter": 30}
        ```

*   **2.3 Funktionen und Module:**
    *   **Beschreibung:** **Funktionen** sind wiederverwendbare Codeblöcke, die eine bestimmte Aufgabe erfüllen [2.3, 56]. **Module** sind Dateien, die Python-Definitionen und Anweisungen enthalten und es ermöglichen, Code zu organisieren und wiederzuverwenden [2.3, 56]. Module werden mit dem Schlüsselwort `import` eingebunden [2.3]. Es ist üblich, Bibliotheken wie `numpy`, `pandas` und `matplotlib.pyplot` mit Aliasen zu importieren (z.B. `import matplotlib.pyplot as plt`) [2.3, 65, 91]. Die **Tab-Vervollständigung** in IPython und Jupyter Notebooks kann beim Importieren und Erkunden von Objekten hilfreich sein [2.3, 89, 90, 92].
    *   **Codebeispiel (Funktionsdefinition und Modulimport):**
        ```python
        # Funktionsdefinition
        def gruss(name):
            print(f"Hallo, {name}!")

        # Modulimport
        import math
        print(math.sqrt(16))

        # Modulimport mit Alias
        import pandas as pd
        df = pd.DataFrame({'A':, 'B':})
        print(df)

        # Importieren spezifischer Funktionen aus einem Modul
        from datetime import datetime
        print(datetime.now())
        ```

*   **2.4 Dateiverarbeitung:**
    *   **Beschreibung:** Python ermöglicht das Lesen und Schreiben von Dateien [2.4, 111, 112, 139, 140]. Dateien können im **Textmodus** (`'r'`, `'w'`, `'a'`, `'r+'`, `'t'`) oder im **Binärmodus** (`'rb'`, `'wb'`, `'b'`) geöffnet werden. Beim Lesen von Textdateien werden Zeilenendezeichen (`\n`) beibehalten. Es ist wichtig, Dateien nach der Verarbeitung explizit zu schließen, um Ressourcen freizugeben. Die `with`-Anweisung bietet eine bequeme Möglichkeit, Dateien zu öffnen und automatisch zu schließen. Die Methoden `read()`, `readline()`, `readlines()` werden zum Lesen und `write()`, `writelines()` zum Schreiben verwendet.
    *   **Codebeispiel (Lesen aus einer Textdatei):**
        ```python
        with open('beispiel.txt', 'r') as f:
            for line in f:
                print(line.rstrip()) # Entfernt das Zeilenendezeichen
        ```
    *   **Codebeispiel (Schreiben in eine Textdatei):**
        ```python
        meine_daten = ["Zeile 1\n", "Zeile 2\n", "Zeile 3\n"]
        with open('ausgabe.txt', 'w') as f:
            f.writelines(meine_daten)
        ```

## 3. Datenmanipulation mit Pandas

*   **3.1 Einführung in Pandas DataFrames und Series:**
    *   **Beschreibung:** **Pandas** ist eine zentrale Bibliothek für die Datenanalyse in Python [3.1, 19, 40]. Sie baut auf der NumPy-Bibliothek auf und erweitert diese um leistungsstarke Datenstrukturen wie **DataFrames** und **Series**. Ein **DataFrame** kann als eine tabellenartige Datenstruktur mit beschrifteten Zeilen und Spalten betrachtet werden [3.1, 19, 92]. Eine **Series** ist eine eindimensionale beschriftete Datenstruktur. Pandas-Objekte können als erweiterte Versionen von NumPy Structured Arrays mit Label-Indizes betrachtet werden.
    *   **Codebeispiel (Erstellen einer Pandas Series):**
        ```python
        import pandas as pd
        s = pd.Series(, index=['a', 'b', 'c', 'd'])
        print(s)
        ```
    *   **Codebeispiel (Erstellen eines Pandas DataFrame):**
        ```python
        data = {'Name': ['Alice', 'Bob', 'Charlie'],
                'Alter':,
                'Stadt': ['New York', 'London', 'Paris']}
        df = pd.DataFrame(data)
        print(df)
        ```
        Man kann auch direkt in Python mit der `help()` Funktion die Dokumentation zu Pandas-Objekten abrufen.

*   **3.2 Daten laden, speichern und Dateiformate:**
    *   **Beschreibung:** Pandas bietet Funktionen zum Lesen von Daten aus verschiedenen Dateiformaten wie **CSV** (`pd.read_csv()` [3.2, 9, 73, 142]), **JSON**, Excel und anderen [3.2, 116, 124, 141]. Die Funktion `pd.read_csv()` verfügt über zahlreiche Optionen zur Anpassung des Leseprozesses. Ebenso können Pandas DataFrames in verschiedene Formate gespeichert werden (z.B. `df.to_csv()`).
    *   **Codebeispiel (Lesen einer CSV-Datei):**
        ```python
        import pandas as pd
        df = pd.read_csv('daten.csv')
        print(df.head()) # Zeigt die ersten Zeilen des DataFrames
        ```
    *   **Codebeispiel (Schreiben in eine CSV-Datei):**
        ```python
        import pandas as pd
        data = {'Produkt': ['A', 'B', 'C'], 'Preis':}
        df = pd.DataFrame(data)
        df.to_csv('produkte.csv', index=False) # Speichert den DataFrame ohne den Index
        ```

*   **3.3 Datenbereinigung und -vorbereitung (Data Wrangling):**
    *   **Beschreibung:** **Data Wrangling** (oder Munging) umfasst das Identifizieren und Beheben von Problemen in Datensätzen, wie z.B. fehlende Werte, Duplikate und Inkonsistenzen [3.3, 8, 191]. Pandas bietet Werkzeuge zum **Filtern** von Daten, zum **Umgang mit fehlenden Werten** (`dropna()`, `fillna()` ), zum **Entfernen von Duplikaten** (`drop_duplicates()`) und zur **Typkonvertierung** (`astype()`). Das Bereinigen numerischer Spalten ist ein wichtiger Schritt.
    *   **Codebeispiel (Umgang mit fehlenden Werten):**
        ```python
        import pandas as pd
        data = {'A': [1, 2, None], 'B': [4, None, 6]}
        df = pd.DataFrame(data)
        print(df.isnull()) # Zeigt, welche Werte NaN sind
        df_ohne_na = df.dropna() # Entfernt Zeilen mit NaN-Werten
        df_gefuellt = df.fillna(0) # Füllt NaN-Werte mit 0
        ```
    *   **Codebeispiel (Duplikate entfernen):**
        ```python
        import pandas as pd
        data = {'A':, 'B':}
        df = pd.DataFrame(data)
        df_ohne_duplikate = df.drop_duplicates()
        print(df_ohne_duplikate)
        ```

*   **3.4 Datentransformation:**
    *   **Beschreibung:** **Datentransformation** beinhaltet die Konvertierung von Daten von einem Format oder einer Struktur in ein anderes [3.4, 70, 93]. Dazu gehören Operationen wie das **Hinzufügen neuer Spalten**, das **Umbenennen von Spalten** (`rename()`), das **Anwenden von Funktionen auf Spalten** (`apply()`), das **Erstellen von Dummy-Variablen** (`pd.get_dummies()`) und das **Reshaping** von DataFrames (z.B. mit `pivot_table()`, `stack()`, `unstack()`). Pandas bietet auch **vektorisierte String-Funktionen** für die Bearbeitung von Textdaten.
    *   **Codebeispiel (Hinzufügen einer neuen Spalte):**
        ```python
        import pandas as pd
        data = {'Preis':}
        df = pd.DataFrame(data)
        df['MwSt'] = df['Preis'] * 0.19
        print(df)
        ```
    *   **Codebeispiel (Anwenden einer Funktion):**
        ```python
        import pandas as pd
        def rabatt(preis):
            if preis > 15:
                return preis * 0.9
            else:
                return preis
        data = {'Preis':}
        df = pd.DataFrame(data)
        df['Preis_mit_Rabatt'] = df['Preis'].apply(rabatt)
        print(df)
        ```

*   **3.5 Datenmerging und -abfragen:**
    *   **Beschreibung:** Pandas ermöglicht das **Zusammenführen (Merging)** von DataFrames basierend auf gemeinsamen Spalten (`pd.merge()`) und das **Verketten (Concatenating)** von DataFrames entlang einer Achse (`pd.concat()`) [3.5, 144]. Das **Abfragen (Querying)** von Daten erfolgt durch Filtern von Zeilen basierend auf Bedingungen.
    *   **Codebeispiel (Mergen von DataFrames):**
        ```python
        import pandas as pd
        df1 = pd.DataFrame({'ID':, 'Name': ['A', 'B', 'C']})
        df2 = pd.DataFrame({'ID':, 'Wert':})
        merged_df = pd.merge(df1, df2, on='ID', how='inner') # Inner Join basierend auf der Spalte 'ID'
        print(merged_df)
        ```
    *   **Codebeispiel (Abfragen von Daten):**
        ```python
        import pandas as pd
        data = {'Produkt': ['A', 'B', 'A', 'C'], 'Preis':}
        df = pd.DataFrame(data)
        teure_produkte = df[df['Preis'] > 12]
        print(teure_produkte)
        ```

*   **3.6 Umgang mit Zeitreihendaten:**
    *   **Beschreibung:** Pandas verfügt über spezielle Funktionen für die Arbeit mit **Zeitreihendaten** [3.6, 15, 38, 169, 171]. Dies umfasst das Erstellen von Datums- und Zeitreihen-Indizes, das **Resampling** von Zeitreihendaten (`resample()`), das **Verschieben von Zeitreihen** (`shift()`) und die Arbeit mit **Zeitreihenfrequenzen**, die durch Offset-Aliase dargestellt werden können [3.6, 15].
    *   **Codebeispiel (Erstellen einer Zeitreihen-Series):**
        ```python
        import pandas as pd
        dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
        ts = pd.Series(, index=dates)
        print(ts)
        ```
    *   **Codebeispiel (Resampling von Zeitreihendaten):**
        ```python
        import pandas as pd
        dates = pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 10:30:00', '2023-01-01 11:00:00'])
        data =
        ts = pd.Series(data, index=dates)
        # Durchschnittliche Werte pro Stunde berechnen
        hourly_avg = ts.resample('H').mean()
        print(hourly_avg)
        ```

## 4. Datenvisualisierung mit Matplotlib und Seaborn

*   **4.1 Einführung in Matplotlib:**
    *   **Beschreibung:** **Matplotlib** ist eine grundlegende Bibliothek für die **Datenvisualisierung** in Python [4.1, 57, 65, 67, 81, 91, 145, 170]. Das Submodul `pyplot` wird üblicherweise als `plt` importiert und bietet eine Sammlung von Funktionen, die Matplotlib wie MATLAB funktionieren lassen [4.1, 57, 65, 91, 145]. Die Verwendung des Magic Commands `%matplotlib inline` in Jupyter Notebooks ermöglicht die Anzeige von Plots direkt im Notebook. Matplotlib bietet eine große Auswahl an anpassbaren Plottypen und Backends.
    *   **Codebeispiel (Einfacher Linienplot mit Matplotlib):**
        ```python
        import matplotlib.pyplot as plt
        x =
        y =
        plt.plot(x, y)
        plt.xlabel('X-Achse')
        plt.ylabel('Y-Achse')
        plt.title('Einfacher Linienplot')
        plt.show()
        ```

*   **4.2 Erstellen verschiedener Diagrammtypen (Linien-, Balken-, Streudiagramme etc.):**
    *   **Beschreibung:** Matplotlib ermöglicht die Erstellung verschiedenster **Diagrammtypen**, darunter **Liniendiagramme** (`plt.plot()`), **Balkendiagramme** (`plt.bar()` oder `plt.barh()` für horizontale Balken), **Streudiagramme** (`plt.scatter()`), **Histogramme** (`plt.hist()`), **Kuchendiagramme** (`plt.pie()`) und mehr [4.2, 68, 146]. Die Wahl des geeigneten Diagrammtyps ist ein wichtiger Bestandteil der Datenvisualisierung.
    *   **Codebeispiel (Balkendiagramm mit Matplotlib):**
        ```python
        import matplotlib.pyplot as plt
        produkte = ['A', 'B', 'C']
        verkaufszahlen =
        plt.bar(produkte, verkaufszahlen)
        plt.xlabel('Produkt')
        plt.ylabel('Verkaufszahlen')
        plt.title('Verkaufszahlen pro Produkt')
        plt.show()
        ```
    *   **Codebeispiel (Streudiagramm mit Matplotlib):**
        ```python
        import matplotlib.pyplot as plt
        x =
        y =
        plt.scatter(x, y)
        plt.xlabel('Variable X')
        plt.ylabel('Variable Y')
        plt.title('Streudiagramm')
        plt.show()
        ```

*   **4.3 Anpassung und Formatierung von Plots:**
    *   **Beschreibung:** Matplotlib erlaubt die umfassende **Anpassung und Formatierung** von Plots [4.3, 82, 146, 28]. Dazu gehören das Hinzufügen von **Titeln** (`plt.title()` [4.3, 91]), **Achsenbeschriftungen** (`plt.xlabel()`, `plt.ylabel()` [4.3, 91]), **Legenden** (`plt.legend()` [4.3, 146]), das **Ändern von Linienfarben und -stilen**, das **Anpassen von Achsenticks und -grenzen** (`plt.xticks()`, `plt.yticks()`, `plt.xlim()`, `plt.ylim()`), das Hinzufügen von **Gitternetzen** (`plt.grid()`) und **Annotationen** (`plt.annotate()`). Die Formatierung kann auch über Style Sheets und `rcParams` angepasst werden.
    *   **Codebeispiel (Anpassen eines Plots):**
        ```python
        import matplotlib.pyplot as plt
        x =
        y =
        plt.plot(x, y, color='green', linestyle='--', marker='o', label='Daten')
        plt.xlabel('Zeit')
        plt.ylabel('Wert')
        plt.title('Angepasster Linienplot')
        plt.legend()
        plt.grid(True)
        plt.show()
        ```

*   **4.4 Fortgeschrittene Visualisierungen mit Seaborn:**
    *   **Beschreibung:** **Seaborn** ist eine auf Matplotlib aufbauende Bibliothek, die eine höhere Abstraktionsebene für **statistische Visualisierungen** bietet [4.4, 27, 65, 91, 96, 170]. Seaborn integriert sich gut mit Pandas DataFrames und ermöglicht die einfache Erstellung ansprechender und informativer Plots wie **Streudiagramme mit Regressionslinien**, **Boxplots**, **Violinplots**, **Heatmaps** (zur Visualisierung von Korrelationen [5.3]), **Histogramme mit Dichtekurven** und vieles mehr [4.4, 27, 96]. Seaborn bietet auch Funktionen zur Steuerung der Farbpaletten und des allgemeinen Erscheinungsbilds von Plots.
    *   **Codebeispiel (Streudiagramm mit Seaborn):**
        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        data = {'X':, 'Y':}
        df = pd.DataFrame(data)
        sns.scatterplot(x='X', y='Y', data=df)
        plt.title('Streudiagramm mit Seaborn')
        plt.show()
        ```
    *   **Codebeispiel (Boxplot mit Seaborn):**
        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        data = {'Kategorie': ['A', 'A', 'B', 'B', 'A', 'B'], 'Wert':}
        df = pd.DataFrame(data)
        sns.boxplot(x='Kategorie', y='Wert', data=df)
        plt.title('Boxplot mit Seaborn')
        plt.show()
        ```

## 5. Explorative Datenanalyse (EDA) und Statistik

*   **5.1 Grundlagen der explorativen Datenanalyse:**
    *   **Beschreibung:** Die **Explorative Datenanalyse (EDA)** ist ein wichtiger erster Schritt im Datenanalyseprozess [5.1, 8, 9, 11, 67]. Ziel ist es, die **Daten zu verstehen**, **Muster und Beziehungen zu erkennen**, **Hypothesen zu generieren** und potenzielle Probleme oder Anomalien im Datensatz zu identifizieren [5.1, 67]. EDA umfasst oft die Visualisierung von Daten und die Berechnung deskriptiver Statistiken [5.1, 40].
    *   **Codebeispiel (Grundlegende EDA mit Pandas):**
        ```python
        import pandas as pd
        df = pd.read_csv('daten.csv')
        print(df.head())        # Erste Zeilen anzeigen
        print(df.info())        # Informationen über den DataFrame
        print(df.describe())    # Deskriptive Statistiken
        print(df.columns)       # Spaltennamen
        print(df.shape)         # Anzahl der Zeilen und Spalten
        ```

*   **5.2 Deskriptive Statistik:**
    *   **Beschreibung:** **Deskriptive Statistiken** fassen die Haupteigenschaften eines Datensatzes zusammen [5.2, 16, 71, 72]. Dazu gehören **Maße der zentralen Tendenz** (z.B. Mittelwert, Median, Modus) und **Maße der Dispersion** (z.B. Standardabweichung, Varianz, Spannweite, Perzentile) [5.2, 72]. Pandas bietet Funktionen wie `mean()`, `median()`, `mode()`, `std()`, `var()`, `min()`, `max()`, `quantile()` und `describe()` zur Berechnung dieser Statistiken [5.2].
    *   **Codebeispiel (Deskriptive Statistik mit Pandas):**
        ```python
        import pandas as pd
        data = {'Alter':}
        df = pd.DataFrame(data)
        print(df['Alter'].mean())       # Mittelwert
        print(df['Alter'].median())     # Median
        print(df['Alter'].std())        # Standardabweichung
        print(df['Alter'].describe())   # Umfassende deskriptive Statistik
        ```

*   **5.3 Korrelation:**
    *   **Beschreibung:** **Korrelation** misst die Stärke und Richtung der linearen Beziehung zwischen zwei oder mehr Variablen [5.3, 74, 190]. Der **Korrelationskoeffizient** (z.B. Pearson) liegt zwischen -1 und 1. Eine positive Korrelation bedeutet, dass die Variablen tendenziell gleichzeitig steigen oder fallen, während eine negative Korrelation bedeutet, dass eine Variable tendenziell steigt, wenn die andere fällt. **Heatmaps** (erstellt z.B. mit Seaborn) können zur Visualisierung von Korrelationsmatrizen verwendet werden [5.3]. Es ist wichtig zu beachten, dass Korrelation keine Kausalität impliziert.
    *   **Codebeispiel (Korrelation mit Pandas):**
        ```python
        import pandas as pd
        data = {'Umsatz':, 'Werbung':}
        df = pd.DataFrame(data)
        korrelation = df['Umsatz'].corr(df['Werbung'])
        print(f"Korrelation zwischen Umsatz und Werbung: {korrelation}")

        # Korrelationsmatrix für alle numerischen Spalten
        korrelationsmatrix = df.corr()
        print(korrelationsmatrix)

        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.heatmap(korrelationsmatrix, annot=True, cmap='coolwarm')
        plt.title('Korrelationsmatrix')
        plt.show()
        ```

*   **5.4 Hypothesentest und Regression:**
    *   **Beschreibung:** **Hypothesentests** werden verwendet, um statistische Entscheidungen auf Basis experimenteller Daten zu treffen und Annahmen über Populationsparameter zu validieren [5.4, 75, 235]. **Regression** ist eine statistische Methode zur Modellierung der Beziehung zwischen einer abhängigen Variable und einer oder mehreren unabhängigen Variablen [5.4, 75]. Es gibt verschiedene Arten von Regressionen, wie z.B. die **lineare Regression**. Python-Bibliotheken wie SciPy und statsmodels bieten Funktionen für Hypothesentests und Regressionsanalysen.
    *   **Codebeispiel (Einfache lineare Regression mit statsmodels):**
        ```python
        import statsmodels.api as sm
        import pandas as pd
        data = {'X':, 'Y':}
        df = pd.DataFrame(data)
        X = df['X']
        y = df['Y']
        X = sm.add_constant(X) # Fügt eine Konstante für den Intercept hinzu
        model = sm.OLS(y, X).fit()
        print(model.summary())
        ```

## 6. Einführung in Machine Learning

*   **6.1 Überblick über Machine Learning:**
    *   **Beschreibung:** **Machine Learning (ML)** ist ein Bereich, in dem Datenanalyse auf statistisches Denken trifft, um aus Daten zu lernen [6.1, 38, 39, 40, 76]. Der ML-Workflow umfasst typischerweise **Datenvorverarbeitung**, **Datenvorbereitung**, **Modelltraining**, **Modellevaluierung** und potenziell **Modelldepoyment** [6.1, 40]. Es gibt verschiedene Arten von Machine Learning, darunter **überwachtes Lernen**, **unüberwachtes Lernen** und **Reinforcement Learning**.
    *   **Codebeispiel (Grundlegender ML-Workflow mit scikit-learn - konzeptionell):**
        ```python
        # 1. Daten laden und vorbereiten
        import pandas as pd
        from sklearn.model_selection import train_test_split
        # df = pd.read_csv('...')
        # X = df[['feature1', 'feature2']]
        # y = df['zielvariable']
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # 2. Modell auswählen und trainieren
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        # model.fit(X_train, y_train)

        # 3. Modell evaluieren
        # predictions = model.predict(X_test)
        # from sklearn.metrics import mean_squared_error
        # mse = mean_squared_error(y_test, predictions)
        # print(f"Mittlerer quadratischer Fehler: {mse}")
        ```

*   **6.2 Datenvorverarbeitung für Machine Learning:**
    *   **Beschreibung:** Vor dem Training von Machine Learning Modellen ist eine **Datenvorverarbeitung** unerlässlich [6.2, 40, 557]. Dazu gehören Schritte wie **Skalierung von Features** (z.B. Standardisierung, Normalisierung), **Umgang mit kategorialen Variablen** (z.B. One-Hot-Encoding), **Feature Engineering** (Erstellen neuer Features aus bestehenden) und das **Aufteilen der Daten in Trainings- und Testsets** (`train_test_split` aus `sklearn.model_selection`) [6.2, 40]. **Pipelines** in scikit-learn können verwendet werden, um die Vorverarbeitungsschritte und das Modell in einem Workflow zu bündeln.
    *   **Codebeispiel (Datenvorverarbeitung mit scikit-learn):**
        ```python
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import pandas as pd

        data = {'Feature1':, 'Feature2':, 'Ziel':}
        df = pd.DataFrame(data)
        X = df[['Feature1', 'Feature2']]
        y = df['Ziel']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("Skalierte Trainingsdaten:")
        print(X_train_scaled)
        ```

*   **6.3 Beispiele für Machine Learning Modelle:**
    *   **Beschreibung:** Die Quellen erwähnen verschiedene **Anwendungen von Machine Learning**, wie z.B. die Vorhersage der Rotweinqualität, die Klassifizierung von Weinen (rot oder weiß) basierend auf chemischen Eigenschaften und die Erstellung von Regressionsmodellen zur Vorhersage der Jahreslänge von Planeten [6.3, 39]. Bibliotheken wie **scikit-learn (`sklearn`)** werden für Machine Learning in Python verwendet und bieten eine Vielzahl von Algorithmen für Klassifizierung (z.B. Support Vector Machines, Random Forest), Regression und Clustering.
    *   **Codebeispiel (Klassifikation mit scikit-learn):**
        ```python
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        import pandas as pd

        data = {'Feature1':, 'Feature2':, 'Ziel':}
        df = pd.DataFrame(data)
        X = df[['Feature1', 'Feature2']]
        y = df['Ziel']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        genauigkeit = accuracy_score(y_test, predictions)
        print(f"Genauigkeit des Modells: {genauigkeit}")
        