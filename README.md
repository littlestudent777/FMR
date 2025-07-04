# Калькулятор частоты ферромагнитного резонанса (ФМР)

Программа для расчёта частоты ферромагнитного резонанса с учётом влияния:
- внешнего магнитного поля
- размагничивающего поля
- кубической анизотропии
и их комбинаций

## Выражение для угловой частоты
$$\omega_r = \frac{\gamma}{M_s sin(\theta)} (F_{\theta \theta} F_{\phi \phi} - F_{\theta \phi}^2)^{\frac{1}{2}}$$  
(J.Smit, H.G. Beljers), 
где вторая производная берётся в равновесном состоянии.  
Формула не может быть использована при $sin(\theta) \approx 0$.

## Замечание о реализации
Вторые производные уже представлены аналитически. Они получены с помощью инструментов символьного дифференцирования (см. calc_second_derivatives.py).

## Результаты
Для нескольких конфигураций была посчитана частота ФМР по указанной формуле, а также исходя из частоты колебаний одной из поперечных компонент намагниченности, которая была посчитана из численного решения уравнения ЛЛГ (https://github.com/littlestudent777/LLG_equation (функция ‎LLGSolver.compute_frequency)).

| Конфигурация эксперимента               | ФМР по формуле С.-Б. (рад/с) | Угловая частота колебаний компоненты намагниченности (рад/с) |
|-----------------------------------------|---------------------------|---------------------------|
| Внешнее поле (H = [600, 250, 1000] G)   | 2.099e+10                 | 2.101e+10                 |
| Внешнее поле (H = [800, 1000, 350] G)   | 2.337e+10                 | 2.326e+10                 |
| Анизотропия (K1 = 4.2e5, K2 = 1.5e5)    | 8.661e+09                 | 8.654e+09                 |
| Анизотропия (K1 = -4.2e5, K2 = 1.0e5)   | 5.316e+09                 | 5.311e+09                 |
| Размагничивающее поле (цилиндр, x=inf)  | 1.888e+11                 | 1.887e+11                 |
| Комбинация эффектов                     | 7.703e+10                 | 7.701e+10                 |

Результаты сошлись с точностью до первого знака после запятой.
