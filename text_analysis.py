#Creado por Stephen Morales
#Universidad Internacional de la Rioja
#Última actualización: 06/12/2021
#Trabajo de Fin de Máster

from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import requests
from nltk import word_tokenize
import re
import pandas as pd
import numpy as np
import datetime
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from collections import OrderedDict
import gensim.summarization
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer



######################################################### Lista de palabras y simbolos para la limpieza del texto ##########################

palabras_stop = set(stopwords.words('spanish'))
words_stop = set(stopwords.words('english'))

preposiciones = ['a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante','del','ni',
                 'en', 'entre', 'hacia', 'hasta', 'mediante', 'para', 'por', 'segun', 'sin', 'so', 'sobre', 'tras', 'versus', 'via']

pronombres = ['yo', 'tu', 'el', 'usted', 'ustedes', 'nosotros', 'nosotras',
              'vosotros', 'vosotras', 'ellos', 'ellas', 'me', 'te', 'nos', 'se',
             'mi', 'mis', 'mio', 'mia', 'mios', 'mias', 'tu', 'tus', 'tuyo', 'tuya',
              'tuyos', 'tuyas', 'su', 'sus', 'suyo', 'suya', 'suyos', 
              'suyas', 'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra', 'vuestros', 'vuestra',
             'este', 'ese', 'aquel', 'esta', 'esa', 'aquella', 'estos', 'esos', 'aquellos',
              'estas', 'esas', 'aquellas', 'esto', 'eso', 'aquello',
              'que', 'el que', 'la que', 'las que', 'lo que', 'los que', 'quien', 'quienes', 'cual', 
              'cuales', 'cual', 'cuyo', 'cuyos', 'cuyas', 'cuya',
               'cuanto', 'cuantos', 'cuanta', 'cuantas', 'donde', 'como',
              'mucho', 'muchos', 'mucha', 'muchas', 'poco', 'pocos', 'poca', 'pocas', 'tanto', 'tantos', 
              'tanta', 'tantas', 'bastante', 'bastantes',
              'demasiado', 'demasiados', 'demasiada', 'demasiadas', 'alguno', 'algunos', 'alguna', 
              'algunas', 'ninguno', 'ninguna', 'algo', 'nada'

             ]

conectores = ['y', 'ademas', 'tambien', 'asimismo', 'también', 'por', 'añadidura', 'igualmente',
             'encima', 'es', 'mas', 'aún', 'incluso', 'hasta', 'para', 'colmo',
              'con', 'todo', 'pesar', 'todo', 'aun', 'asi', 'cualquier', 'modo', 'al', 'mismo', 
              'pero', 'sin', 'embargo', 'obstante', 'cierto', 'modo', 'cierta', 'cierto', 'otra', 'empero','otro',
              'tanto', 'ende', 'consiguiente',  'consecuencia','pues', 'tanto', 'eso', 'sigue', 'entonces', 'manera',
              'porque', 'pues', 'puesto', 'ya', 'causa', 'visto', 'dado', 'como', 'a','e','i','o','u',
             'modo', 'igualmente', 'analogamente', 'similar','sea', 'esto', 'otras','otros', 'hecho',
              'resumen', 'resumidas', 'cuentas', 'definitiva', 'suma', 'total', 'breve', 'sintesis', 'recapitulando', 
              'brevemente','ejemplo', 'verbigracia', 'particularmente','particular', 'especificamente', 
              'incidentalmente', 'caso' ,'efectivamente','fin', 'ultimo', 'finalmente', 'final', 'resumir', 'conclusion', 
              'finalizar', 'pues', 'definitiva','lado', 'parte', 'continuacion', 'despues', 'luego', 'respecto', 'acerca',
              'cuando','cuanto','entonces', 'quien','donde','sino'
             ]

adverbio = ['aqui', 'alli', 'ahi', 'alla', 'aca', 'arriba', 'abajo', 'cerca', 'lejos', 'adelante',
            'delante', 'detras', 'encima', 'debajo', 'enfrente', 'atras', 'alrededor',
            'antes', 'despues', 'luego', 'pronto', 'tarde', 'temprano', 'todavia', 'aun', 
            'ya', 'ayer', 'hoy', 'mañana', 'anteayer', 'siempre',
            'proximamente', 'prontamente', 'anoche', 'enseguida', 'mientras', 'anteriormente',
            'si', 'tambien', 'cierto', 'ciertamente', 'efectivamente', 'claro', 'exacto', 'obvio', 
            'solo','solamente','inclusive', 'unicamente', 'incluso', 'mismamente', 'propiamente', 'precisamente', 
            'concretamente', 'siquiera', 'consecuentemente',
            'quiza','quizas', 'acaso', 'probablemente', 'posiblemente', 'seguramente', 'tal', 'vez'
           ]

articulo = ['el', 'la', 'lo','las', 'los','uno','una','unos','unas','un' ]

todas = ['a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante','del','ni',
                 'en', 'entre', 'hacia', 'hasta', 'mediante', 'para', 'por', 'segun', 'sin', 'so', 'sobre', 'tras', 'versus', 'via','yo', 'tu', 'el', 'usted', 'ustedes', 'nosotros', 'nosotras',
              'vosotros', 'vosotras', 'ellos', 'ellas', 'me', 'te', 'nos', 'se',
             'mi', 'mis', 'mio', 'mia', 'mios', 'mias', 'tu', 'tus', 'tuyo', 'tuya',
              'tuyos', 'tuyas', 'su', 'sus', 'suyo', 'suya', 'suyos', 
              'suyas', 'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra', 'vuestros', 'vuestra',
             'este', 'ese', 'aquel', 'esta', 'esa', 'aquella', 'estos', 'esos', 'aquellos',
              'estas', 'esas', 'aquellas', 'esto', 'eso', 'aquello',
              'que', 'el que', 'la que', 'las que', 'lo que', 'los que', 'quien', 'quienes', 'cual', 
              'cuales', 'cual', 'cuyo', 'cuyos', 'cuyas', 'cuya',
               'cuanto', 'cuantos', 'cuanta', 'cuantas', 'donde', 'como',
              'mucho', 'muchos', 'mucha', 'muchas', 'poco', 'pocos', 'poca', 'pocas', 'tanto', 'tantos', 
              'tanta', 'tantas', 'bastante', 'bastantes',
              'demasiado', 'demasiados', 'demasiada', 'demasiadas', 'alguno', 'algunos', 'alguna', 
            'algunas', 'ninguno', 'ninguna', 'algo', 'nada','y', 'ademas', 'tambien', 'asimismo', 'también', 'por', 'añadidura', 'igualmente','encima', 'es', 'mas', 'aún', 'incluso', 'hasta', 'para', 'colmo',
              'con', 'todo', 'pesar', 'todo', 'aun', 'asi', 'cualquier', 'modo', 'al', 'mismo', 
              'pero', 'sin', 'embargo', 'obstante', 'cierto', 'modo', 'cierta', 'cierto', 'otra', 'empero','otro',
              'tanto', 'ende', 'consiguiente',  'consecuencia','pues', 'tanto', 'eso', 'sigue', 'entonces', 'manera',
              'porque', 'pues', 'puesto', 'ya', 'causa', 'visto', 'dado', 'como', 'a','e','i','o','u',
             'modo', 'igualmente', 'analogamente', 'similar','sea', 'esto', 'otras','otros', 'hecho',
              'resumen', 'resumidas', 'cuentas', 'definitiva', 'suma', 'total', 'breve', 'sintesis', 'recapitulando', 
              'brevemente','ejemplo', 'verbigracia', 'particularmente','particular', 'especificamente', 
              'incidentalmente', 'caso' ,'efectivamente','fin', 'ultimo', 'finalmente', 'final', 'resumir', 'conclusion', 
              'finalizar', 'pues', 'definitiva','lado', 'parte', 'continuacion', 'despues', 'luego', 'respecto', 'acerca',
              'cuando','cuanto','entonces', 'quien','donde','sino','aqui', 'alli', 'ahi', 'alla', 'aca', 'arriba', 'abajo', 'cerca', 'lejos', 'adelante','delante', 'detras', 'encima', 'debajo', 'enfrente', 'atras', 'alrededor',
            'antes', 'despues', 'luego', 'pronto', 'tarde', 'temprano', 'todavia', 'aun', 
            'ya', 'ayer', 'hoy', 'mañana', 'anteayer', 'siempre',
            'proximamente', 'prontamente', 'anoche', 'enseguida', 'mientras', 'anteriormente',
            'si', 'tambien', 'cierto', 'ciertamente', 'efectivamente', 'claro', 'exacto', 'obvio', 
            'solo','solamente','inclusive', 'unicamente', 'incluso', 'mismamente', 'propiamente', 'precisamente', 
            'concretamente', 'siquiera', 'consecuentemente',
            'quiza','quizas', 'acaso', 'probablemente', 'posiblemente', 'seguramente', 'tal', 'vez','el', 'la', 'lo','las', 'los','uno','una','unos','unas','un' ]

simbolos = [':', ',', '.',';','!','?','¿','¡','\n','-',"_","#","(",")","[","]","{","}","%","&","*","=",'\n\n','\t']

####################################################### Limpieza de Texto ######################################################

def tilde(string):
    
    """Elimina las tildes del texto""" 
    
    string = string.replace("á","a")
    string = string.replace("é","e")
    string = string.replace("í","i")
    string = string.replace("ó","o")
    string = string.replace("ú","u")
    
    return string

def simbolo(string):
    
    """Elimina los simbolos del texto"""
    
    reemplazo = []
    for i in simbolos:
        reemplazo.append((i,' '))
    
    for a, b in reemplazo:
        string = string.replace(a, b)
    return string

def numero(string):
    
    """Elimina los numeros del texto"""
    
    string = filter(lambda w: w.isalpha(), string.split())
    string = tuple(string)
    string = ' '.join(string)
    return string

def eliminacion_palabras(string):
    
    """Elimina las preposiciones, pronombre, conectores, adverbios y articulos del texto"""
    
    string = filter(lambda w: (w not in palabras_stop)&(w not in preposiciones)&(w not in pronombres) & 
           (w not in conectores) & (w not in adverbio) & (w not in articulo), string.split())
    
    string = tuple(string)
    string = ' '.join(string)
    return string

def eliminacion_palabras_en(string):
    
    """Elimina las palabras stop en ingles"""
    
    string = filter(lambda w: (w not in words_stop), string.split())
    
    string = tuple(string)
    string = ' '.join(string)
    return string

def longitud_minima(string, n=2):
    """Escoge las palabras con una longitud mínima mayor a n"""
    palabras_deseadas = []
    for i in string.split(" "):
        if len(i)>n:
            palabras_deseadas.append(i)
    return " ".join(palabras_deseadas)

def limpieza_texto_en(string, n=2, simbolo_no = True):
    
    """
    Recibe un string
    Transforma el texto a minúsculas
    Depura el texto en ingles"""
    
    string = string.lower()
    if simbolo_no == True:
        string = simbolo(string)
    else:
        pass
    string = eliminacion_palabras_en(string)
    string = longitud_minima(string, n)
    
    return string

def limpieza_texto(string, n=2, simbolo_no = True):

    """
    Recibe un string
    Transforma el texto a minúsculas
    Depura el texto de números, símbolos, palabras no útiles y tildes
    """
    string = string.lower()
    if simbolo_no == True:
        string = simbolo(string)
    else:
        pass
    string = numero(string)
    string = tilde(string)
    string = eliminacion_palabras(string)
    string = longitud_minima(string, n)
    
    return string
 
####################################################### Lectura del texto ############################################################

def df_tweets(tweets_dict):
    
    #Guarda los años
    anios = []
    for i in tweets_dict.keys():
        anios.append(i.split("-")[1])
        
    #Guarda los meses
    mes = []
    for i in tweets_dict.keys():
        mes.append(i.split("-")[0]) 
        
    #Guarda el tweet en el Data Frame
    tweets = []
    for i in tweets_dict.keys():
        tweet=""
        for j in tweets_dict[i]:
            tweet = tweet+j
        tweets.append(tweet)
        
    #Guarda el numero de tweets en cada mes
    n_tweets = []
    for i in tweets_dict.keys():
        n_tweets.append(len(tweets_dict[i]))
        
    #Realiza el promedio de conteo de palabras por tweet
    promedio_palabras = []

    for i in tweets_dict.keys():
        conteo_palabras = []
        for j in tweets_dict[i]:
            conteo_palabras.append(contar_palabras(j))
        promedio_palabras.append(np.mean(conteo_palabras))
        
    tweets_df = pd.DataFrame({"año":anios, "Número tweets":n_tweets,"Mes":mes,"Tweet":tweets,"Promedio_Palabras":promedio_palabras})
    
    #Codificamos en números los meses
    tweets_df["N_mes"] = np.where(tweets_df.Mes=="Ene",1,
                        np.where(tweets_df.Mes=="Feb",2,
                        np.where(tweets_df.Mes=="Mar",3,
                        np.where(tweets_df.Mes=="Abr",4,
                        np.where(tweets_df.Mes=="May",5,
                        np.where(tweets_df.Mes=="Jun",6,
                        np.where(tweets_df.Mes=="Jul",7,
                        np.where(tweets_df.Mes=="Ago",8,
                        np.where(tweets_df.Mes=="Sep",9,
                        np.where(tweets_df.Mes=="Oct",10,
                        np.where(tweets_df.Mes=="Nov",11,
                        np.where(tweets_df.Mes=="Dic",12,np.nan))))))))))))
    
    numero_mes = [10,11,12]
    fecha = []
    for i in range(len(tweets_df.N_mes)):
        fecha_string = str((np.where(np.isin(tweets_df.N_mes[i],numero_mes),'01/'+str(int(tweets_df.N_mes[i]))+'/'+str(tweets_df.año[i]),'01/'+"0"+str(int(tweets_df.N_mes[i]))+'/'+str(tweets_df.año[i]))))
        fecha_obj = datetime.datetime.strptime(fecha_string, '%d/%m/%Y')
        fecha.append(fecha_obj)

    tweets_df["Fecha"] = fecha
    
    return tweets_df


def todo_tweets(tweets_persona):
    todos_tweets = []
    fecha2 = []
    fecha1 = []
    
    for i in tweets_persona.keys():
        todos_tweets += tweets_persona[i]
        fecha1.append([i]*len(tweets_persona[i]))
    
    for i in fecha1:
        for j in i:
            fecha2.append(j)
            
    todos_tweets_df = pd.DataFrame({"tweets":todos_tweets, "fecha_json": fecha2})
    todos_tweets_df["fecha"] = todos_tweets_df["fecha_json"].apply(lambda x: fecha(x))
    
    return todos_tweets_df


def fecha(fecha_string):
    fecha_string = fecha_string.split("-")
    dia = "01"
    mes = np.where(fecha_string[0]=="Ene","01",
                        np.where(fecha_string[0]=="Feb","02",
                        np.where(fecha_string[0]=="Mar","03",
                        np.where(fecha_string[0]=="Abr","04",
                        np.where(fecha_string[0]=="May","05",
                        np.where(fecha_string[0]=="Jun","06",
                        np.where(fecha_string[0]=="Jul","07",
                        np.where(fecha_string[0]=="Ago","08",
                        np.where(fecha_string[0]=="Sep","09",
                        np.where(fecha_string[0]=="Oct","10",
                        np.where(fecha_string[0]=="Nov","11",
                        np.where(fecha_string[0]=="Dic","12",np.nan))))))))))))
    anio = str(fecha_string[1])
    fecha1 = dia + "/" + str(mes) + "/" + anio
    return fecha1
     
    
####################################################### Funciones Especiales ###########################################################
    
def nube_palabras(string,n_palabras=200):
    
    """Muestra una nube de palabras con un estilo predefinido
        No realiza un retorno
    
    """
    
    nube = WordCloud(stopwords=STOPWORDS,background_color='white',width=4000,height=4000, max_words=n_palabras).generate(string)
    plt.imshow(nube)
    plt.axis('off')
    plt.show()

def contar_palabras(string):
    
    "Cuenta las palabras"
    
    return len(string.split(" "))

def palabras_frecuentes(string, n):
    try:
        dict_freq = {}
        for i in range(n):
            dict_freq[Counter(string.split(" ")).most_common()[i][0]] = Counter(string.split(" ")).most_common()[i][1]
        return dict_freq
    except:
        print("Error")

def bigram(string):
    """Recibe un string y devuelve una lista con los bigramas del texto """
    string_split = string.split(" ")
    bigrams = []
    for i in range(len(string_split)):
        try: 
            bigram_tuple = (string_split[i],string_split[i+1])
            bigrams.append(" ".join(bigram_tuple))
        except:
            pass
    return bigrams


def three_gram(string):
    """Recibe un string y devuelve una lista con los bigramas del texto """
    string_split = string.split(" ")
    threegrams = []
    for i in range(len(string_split)):
        try: 
            threegram_tuple = (string_split[i],string_split[i+1],string_split[i+2])
            threegrams.append(" ".join(threegram_tuple))
        except:
            pass
    return threegrams

################################################################# Análisis de sentimientos ############################################

# LISTA DE SENTIMIENTOS POSITIVOS Y NEGATIVOS SACADA DEL SIGUIENTE PAPER

#Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews." Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle, Washington, USA, 

pos_words = requests.get('http://ptrckprry.com/course/ssd/data/positive-words.txt').content.decode('latin-1')
pos_word_list = pos_words.split('\n')
i = 0
while i < len(pos_word_list):
    pos_word = pos_word_list[i]
    if ';' in pos_word or not pos_word:
        pos_word_list.pop(i)
    else:
        i+=1
        
neg_words = requests.get('http://ptrckprry.com/course/ssd/data/negative-words.txt').content.decode('latin-1')
neg_word_list = neg_words.split('\n')
i = 0
while i < len(neg_word_list):
    neg_word = neg_word_list[i]
    if ';' in neg_word or not neg_word:
        neg_word_list.pop(i)
    else:
        i+=1
        
def simple_pos(string):
    n_pos = 0
    token = word_tokenize(string)
    for i in token:
        if i in pos_word_list:
            n_pos += 1
    return n_pos/len(token)

def simple_neg(string):
    n_neg = 0
    token = word_tokenize(string)
    for i in token:
        if i in neg_word_list:
            n_neg += 1
    return n_neg/len(token)

def simple_pol(string):
    n_neg = 0
    token = word_tokenize(string)
    for i in token:
        if i in neg_word_list:
            n_neg += 1
            
    n_pos = 0
    for i in token:
        if i in pos_word_list:
            n_pos += 1
                      
    return (n_pos-n_neg)/(n_pos+n_neg)



nrc = requests.get( 'https://raw.githubusercontent.com/sebastianruder/emotion_proposition_store/master/NRC-Emotion-Lexicon-v0.92/NRC_emotion_lexicon_list.txt').content.decode('latin-1')

nrc_list = re.split('\t|\n', nrc)

lista_palabras = [nrc_list[n:n+3] for n in range(0, len(nrc_list), 3)]


dict_sentimientos = {"anger": [], 'anticipation':[],'disgust':[],'fear':[],'joy':[],'negative':[],
         'positive':[],'sadness':[],'surprise':[],'trust':[]}

emocion = {"anger", 'anticipation','disgust','fear','joy','sadness','surprise','trust'}

for sentimiento in dict_sentimientos.keys():
    for palabra in lista_palabras:
        try:
            if palabra[1] == sentimiento and palabra[2] == "1":
                dict_sentimientos[sentimiento].append(palabra[0])
        except:
            pass

def sentiment_NRC(string,sentiment):
    n_sentiment = 0
    token = word_tokenize(string)
    for i in token:
        if i in dict_sentimientos[sentiment]:
            n_sentiment += 1
    return n_sentiment/len(token)


def palabra_sentimiento(df, nombre_columna, palabra):
    
    list_contiene_palabra = []
    for i in df[nombre_columna]:
        if palabra in i:
            list_contiene_palabra.append("SI")
        else:
            list_contiene_palabra.append("NO")
            
    df["contiene_palabra"] = list_contiene_palabra
    
    tweet_con_palabra = ""
    for i in df[df["contiene_palabra"]=="SI"][nombre_columna]:
        tweet_con_palabra+=i
    
    return SentimentIntensityAnalyzer().polarity_scores(tweet_con_palabra)

def max_emotion(string):
    dict_emotions = {}
    for i in emocion:
        dict_emotions[i] = sentiment_NRC(string,i)
    return max(dict_emotions, key=dict_emotions.get)


####################################################### Texto largo ##################################################################


def leer_texto(text):
    with open(text, encoding='utf8') as f:
        contents = f.readlines()
    texto = ""
    for i in contents: 
        texto = texto+i
    return texto
    
    
def resumen_simple(text,n1,n2):

    oracion_conteo = {}
    
    text = text.lower()
    text = eliminacion_palabras(text)
    
    pf = palabras_frecuentes(text, n1)        

    palabras_mas_frecuentes = [(i, pf[i]) for i in pf.keys()]   

    text2 = text.replace('\n', ' ')

    oraciones = sent_tokenize(text2)            

    for i in oraciones:
        count = 0
        for palabra, frecuencia in palabras_mas_frecuentes:
            if palabra in i:
                count += frecuencia
                oracion_conteo[i] = count
                
    resumen = OrderedDict(sorted(
                        oracion_conteo.items(),
                        key = lambda x: x[1],
                        reverse = True)[:n2])
    return resumen  


def resumen_grafo(text, n=100):
    text = text.replace('\n\n', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    resumen = gensim.summarization.summarize(text, word_count=n) 
    return resumen


############################################## Clustering ################################################################

def elbow(tweet_json,rs = 0):
    
    tweets_df = todo_tweets(tweet_json)
    vectorizer = TfidfVectorizer(stop_words={'spanish'})
    tweets_df["Tweets_Limpios"] = tweets_df.tweets.apply(lambda x: limpieza_texto(x))
    
    documentos = vectorizer.fit_transform(tweets_df.Tweets_Limpios)
    distancias_cuadradas = []
    K = range(2,10)
    for k in K:
        km = KMeans(n_clusters=k, max_iter=100, n_init=1, random_state = rs ) 
        km = km.fit(documentos)
        distancias_cuadradas.append(km.inertia_)
    plt.plot(K, distancias_cuadradas, 'bx-')
    plt.xlabel('k')
    plt.ylabel('distancias_cuadradas')
    plt.show()

def clustering(tweet_json,n_cluster, rs = 0): 
   
    tweets_df = todo_tweets(tweet_json)
    vectorizer = TfidfVectorizer(stop_words={'spanish'})
    tweets_df["Tweets_Limpios"] = tweets_df.tweets.apply(lambda x: limpieza_texto(x))
    
    documentos = vectorizer.fit_transform(tweets_df.Tweets_Limpios)
    
    model = KMeans(n_clusters= n_cluster, max_iter=100, n_init=1, random_state=rs)
    model.fit(documentos)
    
    centroides = model.cluster_centers_.argsort()[:, ::-1]
    palabras = vectorizer.get_feature_names()
    for i in range(n_cluster):
        print("Cluster %d:" % i),
        for j in centroides[i, :10]:
            print(' %s' % palabras[j]),

    tweets_df['Cluster'] = list(model.labels_)
    
    return tweets_df


