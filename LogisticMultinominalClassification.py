# 로지스틱 분류
# 극단치에 따라 기준값이 달라지는걸 예방하기 위해 시그모이드 사용
# 성별, 나이, 페이스를 학습하여 상위, 중위, 하위 그룹 중 분류
from tkinter import *
import tkinter.scrolledtext as tkst
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

# Load the CSV files "marathon_results_2015 ~ 2017.csv" under "data" folder
marathon_2015_2017_qualifying = pd.read_csv("./data/marathon_2015_2017_qualifying.csv")
marathon_2015_2017_qualifying["Grade"] = 1
statistics_2015_2017 = marathon_2015_2017_qualifying["Official Time"].describe()
'''
count    79638.000000
mean     13989.929167
std       2492.272069
min       7757.000000
25%      12258.000000
50%      13592.000000
75%      15325.000000
max      37823.000000
Name: Official Time, dtype: float64
'''
# Official Time 기준 Grade 상위 : 0, 중위 : 1, 하위 : 2
marathon_2015_2017_qualifying.loc[marathon_2015_2017_qualifying["Official Time"] < statistics_2015_2017["25%"], "Grade"] = 0
marathon_2015_2017_qualifying.loc[marathon_2015_2017_qualifying["Official Time"] > statistics_2015_2017["75%"], "Grade"] = 2

# record = pd.DataFrame(marathon_2015_2017,columns=['5K',  '10K',  '15K',  '20K', 'Half',  '25K',  '30K',  '35K',  '40K',  'Official Time']).sort_values(by=['Official Time'])
# 2015~2016년(학습용), 2017년(테스트용) 분류
marathon_2015_2016 = marathon_2015_2017_qualifying[marathon_2015_2017_qualifying['Year'] != 2017]
marathon_2017 = marathon_2015_2017_qualifying[marathon_2015_2017_qualifying['Year'] == 2017]

# 필요한 열만 추출 후 DataFrame 제작
df_2015_2016 = pd.DataFrame(marathon_2015_2016,columns=['M/F',  'Age',  'Pace',  'Grade'])
df_2017 = pd.DataFrame(marathon_2017,columns=['M/F',  'Age',  'Pace',  'Grade'])

# Dataframe to List
record_2015_2016 = df_2015_2016.values.tolist()
record_2017 = df_2017.values.tolist()

nb_classes = 3  # Grade : 0 ~ 2
gender_list = ['Female', 'Male']
grade_list = ['Outstanding(>25%)', 'Average(25~75%)', 'Below(<75%)']

grad_fig = Figure(figsize=(10, 6), dpi=100)
grad_ax = grad_fig.add_subplot(111)
grad_ax.set_xlim(15, 88)
grad_ax.set_ylim(0, 1300)
grad_ax.set_ylabel("Pace : Runner's overall minute per mile pace")
grad_ax.set_xlabel("Age : Age on race day")

def seconds_to_hhmmss(seconds):
    hours = seconds // (60*60)
    seconds %= (60*60)
    minutes = seconds // 60
    seconds %= 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds)

def normalization(record):
    r0 = record[0] # 성별
    r1 = record[1] / 10 # 나이
    r2 = record[2] / 100 # 페이스
    return [r0, r1, r2]

# X and Y data from 0km to 30km
# x_train : [성별, 나이, 페이스], y_train : [Grade]
# normalization : 평균 0, 분산 1로 정규화(normalize)해 네트워크의 학습이 잘 일어나도록 돕는 방식
x_train = [ normalization(r[0:3]) for r in record_2015_2016]
y_train = [ [r[-1]] for r in record_2015_2016]

x_test = [ r[0:3] for r in record_2017]
y_test = [ [r[-1]] for r in record_2017]

def histogram():
    t_a = int(t_aSpbox.get()) - 1
    runner = x_test[t_a] # 해당 선수
    print(runner)
    t_g = int(runner[0]) # 해당 선수의 성별
    t_y = int(runner[1]) # 해당 선수의 나이
    t_p = int(runner[2]) # 해당 선수의 페이스
    if(t_g):
        gender_color = 'b'
    else:
        gender_color = 'r'
    gender_record = df_2017[df_2017['M/F'] == t_g] # 해당 선수와 같은 성별 선수들 추출
    gender_age_record = gender_record[gender_record.Age == t_y]  # 해당 선수와 같은 나이 선수들 추출
    gender_age_record_list = gender_age_record.values.tolist() 
    
    grad_ax.plot(gender_record.Age, gender_record.Pace, '.', color=gender_color, alpha=0.5) # 해당 선수와 성별, 나이가 같은 선수들 표시
    grad_ax.plot(t_y, t_p, 'yd') # 해당 선수 표시
    stat = gender_age_record['Pace'].describe()
    print(stat)
    title = 'Gender : '+gender_list[t_g]+', Age : '+str(t_y)+', Pace : '+str(t_p)
    grad_ax.set_title(title)
    grad_ax.annotate('['+gender_list[t_g]+', '+str(t_y)+']', (75, 1200), fontsize=10)
    grad_ax.annotate("%10s %7i" % ('Count : ', stat[0]), (75, 1150), fontsize=10)
    grad_ax.annotate("%10s %7.3f" % ('Mean :  ', stat[1]), (75, 1100), fontsize=10)
    grad_ax.annotate("%10s %7.3f" % ('25% :   ', stat[3]), (75, 1050), fontsize=10)
    grad_ax.annotate("%10s %7.3f" % ('75% :   ', stat[5]), (75, 1000), fontsize=10)
        
    grad_fig.canvas.draw()    
    
def learing(): 
    """
    MAchine Learning, Kers 
    """
    # Keras related libraries
    import numpy as np
    from keras import optimizers
    from keras.layers import Dense, Activation
    from keras.models import Sequential
    from keras.utils import to_categorical # 원핫코딩
    
    t_a = int(t_aSpbox.get()) - 1
    runner = x_test[t_a]
    t_g = int(runner[0]) # 해당 선수의 성별
    t_y = int(runner[1]) # 해당 선수의 나이
    t_p = int(runner[2]) # 해당 선수의 페이스

    t_t = int(t_tSpbox.get()) # Number of train
    t_r = float(t_rSpbox.get()) # Learning rate

    # Define Sequential model and Dense
    model = Sequential()
    # 3개로 분류 (nb_classes = 3)
    model.add(Dense(nb_classes, input_shape=(3,)))
    # Use softmax : 전체가 1이 되게끔 각 노드의 결과 수정
    model.add(Activation('softmax'))
    # Stochastic gradient descent (SGD) Optimizer
    sgd = optimizers.SGD(lr=t_r)
    # categorical_crossentropy : 각 노드의 결과를 가지고 0과 1로 분류, loss function
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    # prints summary of the model to the terminal
    model.summary()

    # One hot encode [0, 1, 2] to [[1,0,0], [0,1,0], [0,0,1]]
    y_one_hot = to_categorical(y_train)
    print(y_one_hot)
    # Train the model
    history = model.fit(np.array(x_train), np.array(y_one_hot), epochs=t_t)

    # Fit the line
    log_ScrolledText.insert(END, '\nGender :'+gender_list[t_g]+', Age :'+str(t_y)+', Pace :'+str(t_p)+'\n', 'TITLE')
    log_ScrolledText.insert(END, '\n\nCost Decent\n\n','HEADER')
    log_ScrolledText.insert(END, "%10s %20s %20s" % ('Step', 'Cost', 'Accuracy(%)')+'\n\n')
    for step in range(t_t):
        if step % 100 == 0:
            cost_val = history.history['loss'][step]
            a_val = history.history['accuracy'][step]
            log_ScrolledText.insert(END, "%10i %20.5f %20.7f" % (step, cost_val, a_val*100)+'\n')

    # Testing our model, normalization : scale을 맞춰주기 위해
    winner = normalization([ t_g, t_y, t_p ])
    # Get result from the model, 해당 선수의 성별, 나이, 페이스를 입력하여 그룹 예측
    result = model.predict(np.array([winner]))
    # Get one hot index from the model or Use np.argmax(result)
    # predict_classes : 예측결과를 원핫코딩 방식이 아닌 위치값(0,1,2)으로 표시, np.argmax(result) 방법으로도 가능
    grade_index = model.predict_classes(np.array([winner]))
    grade = grade_list[grade_index[0]]
    log_ScrolledText.insert(END, '\n\n')
    log_ScrolledText.insert(END, "%30s" % ('One Hot & Grade Prediction  \n\n'), 'HEADER')
    log_ScrolledText.insert(END, "%30s" % (result))             
    log_ScrolledText.insert(END, '\n\n')
    if(grade_index[0]):
        log_ScrolledText.insert(END, "%30s" % (grade+'\n\n'), 'DisQualifier')
    else:
        log_ScrolledText.insert(END, "%30s" % (grade+'\n\n'), 'Qualifier')       
            
#main
main = Tk()
main.title("Logistic Regression Multinominal Classification")
main.geometry()

label=Label(main, text='Logistic Regression Multinominal Classification')
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=6)

t_aLabel=Label(main, text='Rank of runner : ')
t_aLabel.grid(row=1,column=0)
t_aVal  = IntVar(value=1)
t_aSpbox = Spinbox(main, textvariable=t_aVal ,from_=0, to=len(x_test), increment=1, justify=RIGHT)
#t_aSpbox.config(state='readonly')
t_aSpbox.grid(row=1,column=1)

t_tLabel=Label(main, text='Number of train : ')
t_tLabel.grid(row=1,column=2)
t_tVal  = IntVar(value=10000)
t_tSpbox = Spinbox(main, textvariable=t_tVal ,from_=0, to=100000, increment=1000, justify=RIGHT)
#t_tSpbox.config(state='readonly')
t_tSpbox.grid(row=1,column=3)

t_rLabel=Label(main, text='Learning rate : ')
t_rLabel.grid(row=1,column=4)
t_rVal  = DoubleVar(value=1e-2)
t_rSpbox = Spinbox(main, textvariable=t_rVal ,from_=0, to=1, increment=1e-2, justify=RIGHT)
#t_rSpbox.config(state='readonly')
t_rSpbox.grid(row=1,column=5)

Button(main,text="Histogram", height=2,command=lambda:histogram()).grid(row=2, column=0, columnspan=3, sticky=(W, E))
Button(main,text="Prediction", height=2,command=lambda:learing()).grid(row=2, column=3, columnspan=3, sticky=(W, E))

grad_canvas = FigureCanvasTkAgg(grad_fig, main)
grad_canvas.get_tk_widget().grid(row=3,column=0,columnspan=6)

log_ScrolledText = tkst.ScrolledText(main, height=15)
log_ScrolledText.grid(row=4,column=0,columnspan=6, sticky=(N, S, W, E))
log_ScrolledText.configure(font='TkFixedFont')
log_ScrolledText.tag_config('Qualifier', foreground='blue', font=("Helvetica", 16))
log_ScrolledText.tag_config('DisQualifier', foreground='red', font=("Helvetica", 16))
log_ScrolledText.tag_config('HEADER', foreground='gray', font=("Helvetica", 14), underline=1)
log_ScrolledText.tag_config('TITLE', foreground='orange', font=("Helvetica", 18), underline=1, justify='center')

main.mainloop()