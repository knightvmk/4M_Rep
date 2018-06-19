#include <stdlib.h>
#include <stdio.h>

#include <math.h>
#include <locale.h>
#include <iostream>
#include <Windows.h>
#include <vector>
#include <utility>
#include <omp.h>
using namespace std;
#define scanf scanf_s




inline long double MainTask(const long double &_x, const long double &_y) //1 - x^2 -y^2
{
	return (1 - pow(_x, 2) - pow(_y, 2));
}


inline long double M1(const long double &_a, const long double &_y)//U(a,y)=M1(y)
{
	return (1 - pow(_a, 2) - pow(_y, 2));
}
inline long double M2(const long double &_b, const long double &_y)//U(b,y)=M2(y)
{
	return (1 - pow(_b, 2) - pow(_y, 2));
}
inline long double M3(const long double &_c, const long double &_x)//U(x,a)=M3(x)
{
	return (1 - pow(_x, 2) - pow(_c, 2));
}
inline long double M4(const long double &_d, const long double &_x)//U(x,b)=M4(x)
{
	return (1 - pow(_x, 2) - pow(_d, 2));
}

void InitializingRightSide(long double **B, const long double a, const long double b, const long double c, const long double d, int  n, int m)
{
	long double h = (b - a) / n;
	long double k = (d - c) / m;
	long double x;
	long double y;
	long double f = -4.L;
#pragma omp parallel for
	for (register int j = 1; j < m; j++)
		for (register int i = 1; i < n; i++)
		{
			long double Xi, Yj, sum = 0.L;
			Xi = a + i * h;
			Yj = c + j * k;

			if (j == 1)
			{
				sum += (1 / (k * k)) * M3(c, Xi);
			}
			else
			{
				if (j == m - 1)
				{
					sum += (1 / (k * k)) * M4(d, Xi);
				}
			}
			if (i == 1)
			{
				sum += (1 / (h * h)) * M1(a, Yj);
			}
			else
			{
				if (i == n - 1)
				{
					sum += (1 / (h * h)) * M2(b, Yj);
				}
			}
			B[i][j] = f - sum;
		}
}
double DiscrepancyOfSolution(long double **V, const int n, const int m, const long double a, const long double b, const long double c, const long double d, long double **R)
{
	long double	a2, k2, h2;			//Ненулевые элементы матрицы
	long double  h, k;				//Шаги сетки
	long double** F = new long double*[n + 1];					//Правая часть СЛАУ
	for (register int i = 0; i < m + 1; i++)
	{
		F[i] = new long double[m + 1];
	}

	long double rs = 0.L;				//Невязка

	h = (b - a) / n;
	k = (d - c) / m;

	h2 = ((n / (b - a)) * (n / (b - a)));
	k2 = ((m / (d - c)) * (m / (d - c)));
	a2 = -2 * (h2 + k2);
	//Заполнение вектора правой части(Работает правильно, проверено на тестовой задаче)
	InitializingRightSide(F, a, b, c, d, n, m);
#pragma omp parallel for 
	for (int j = 1; j < m; j++)
	{
		for (int i = 1; i < n; i++)
		{
			double r;
			double mult;

			if (j != 1 && j != m - 1)
			{
				//Внутри блоков
				if (i != 1 && i != n - 1)
					mult = k2 * V[i][j - 1] + h2 * V[i - 1][j] + a2 * V[i][j] + h2 * V[i + 1][j] + k2 * V[i][j + 1];
				else
					if (i == 1)
						mult = k2 * V[i][j - 1] + a2 * V[i][j] + h2 * V[i + 1][j] + k2 * V[i][j + 1];
					else
						if (i == n - 1)
							mult = k2 * V[i][j - 1] + h2 * V[i - 1][j] + a2 * V[i][j] + k2 * V[i][j + 1];
			}
			else
				if (j == 1)//В первом блоке
				{
					if (i == 1)
						mult = a2 * V[i][j] + h2 * V[i + 1][j] + k2 * V[i][j + 1];
					else
						if (i != n - 1)
							mult = h2 * V[i - 1][j] + a2 * V[i][j] + h2 * V[i + 1][j] + k2 * V[i][j + 1];
						else
							if (i == n - 1)
								mult = h2 * V[i - 1][j] + a2 * V[i][j] + k2 * V[i][j + 1];
				}
				else
					if (j == m - 1)//В последнем блоке
					{
						if (i == 1)
							mult = k2 * V[i][j - 1] + a2 * V[i][j] + h2 * V[i + 1][j];
						else
							if (i != n - 1)
								mult = k2 * V[i][j - 1] + h2 * V[i - 1][j] + a2 * V[i][j] + h2 * V[i + 1][j];
							else
								if (i == n - 1)
									mult = k2 * V[i][j - 1] + h2 * V[i - 1][j] + a2 * V[i][j];
					}

			r = fabs(mult - F[i][j]);
			R[i][j] = r;
#pragma omp critical
			if (r > rs)
				rs = r;
		}
	}

	return rs;
}
inline void StartValue(long double **_A, const int _n, const int _m, const long double _a, const long double _b, const long double _c, const long double _d)
{
	long double h = 0.;
	long double k = 0.;
	long double x = 0.;
	long double y = 0.;
	h = (_b - _a) / _n;
	k = (_d - _c) / _m;
#pragma omp parallel for firstprivate(x,y)
	for (register int i = 0; i <= _n; i++)
	{
		for (register int j = 0; j <= _m; j++)
		{
			x = _a + i*h;
			y = _c + j*k;
			if (i == 0)
			{
				_A[i][j] = M1(_a, y);
			}
			if (i == _n)
			{
				_A[i][j] = M2(_b, y);
			}
			if (j == 0)
			{
				_A[i][j] = M3(_c, x);
			}
			if (j == _m)
			{
				_A[i][j] = M4(_d, x);
			}
			if (i != 0 && i != _n && j != 0 && j != _m)
			{
				_A[i][j] = 0.;
			}
		}
	}
}
inline void MethodZeidel(const int print_time, const int _max, long double **_A, const long double _a, const long double _b, const long double _c, const long double _d, const int _n, const int _m, int &S, long double &eps_max, long double &eps_cur, long double &eps)
{
	long double** R = new long double*[_n + 1];					//Правая часть СЛАУ
	for (register int i = 0; i < _m + 1; i++)
	{
		R[i] = new long double[_m + 1];
	}
	short Nmax = _max;
	long double a2, k2, h2;
	long double v_old = 0.;
	long double v_new = 0.;
	bool flag = false;
	h2 = pow((_n / (_b - _a)), 2);
	k2 = pow((_m / (_d - _c)), 2);
	a2 = -2 * (h2 + k2);
	double f = 4;
	double r;
	double Lambda_Max;//используем для оценки
	double Lambda_Min;
	std::vector<std::pair<double, double>> lamb(_n*_m);
	// получаем лямбды
	for (int i = 0, j = 0; i < _n*_m; i++, j++)
	{
		//|z-aii| <= summ(aij)j=1,i!=j
		lamb[i].first = a2; // центр окружности
		double summ = 0;
		if (i != 0 && i != _n*_m - 1)
		{
			summ = fabs(2 * h2) + fabs(k2); // сумма модулей элементов строки за исключением i==j // радиус круга
		}
		else
		{
			summ = fabs(h2) + fabs(k2); // сумма модулей элементов строки за исключением i==j // радиус круга
		}

		lamb[i].second = summ;
	}
	// получили лямбды
	// считаем минимакс лямбды
	// необходимо программно нарисовать круги и определить минимакс
	struct circle
	{
		double left;
		double center;
		double right;
	}; // можно считать по постановке задачи, что значения действительные, расчеты и работа упрощается с кругами
	   // будем считать, что все круги имеют пересечение как минимум попарное (нет пробелов между кругами), иначе хуепутало будет, а не круги Гершговина
	vector<circle> points(_n*_m);
	Lambda_Max = Lambda_Min = lamb[0].first - lamb[0].second;
	for (int i = 0; i < _n + 1; i++)
	{
		points[i].left = lamb[i].first - lamb[i].second;
		points[i].center = lamb[i].first;
		points[i].right = lamb[i].first + lamb[i].second;
		if (points[i].left < Lambda_Min) Lambda_Min = points[i].left;
		if (points[i].right > Lambda_Max) Lambda_Max = points[i].right;
	}
	// получили минимаксы

	// нужны "тао" C (0 , 2 / lamb_max );
	double tao = 2 / Lambda_Max; // не всегда Lambda_i > 0
	double tao_opt = 2 / (Lambda_Max + Lambda_Min);
	double uA = fabs(Lambda_Max) / fabs(Lambda_Min); // мож понадобится // число обусловленности для А матрицы
	double tau = tao_opt;
	while (!flag)
	{
		eps_max = 0;
		for (register int j = 1; j < _m; j++)
			for (register int i = 1; i < _n; i++)
			{
				r= DiscrepancyOfSolution(_A, _n, _m, _a, _b, _c, _d, R);
				v_old = _A[i][j];
				v_new = -tau*R[i][j]+_A[i][j];
				eps_cur = fabs(v_old - v_new);
				if (eps_cur > eps_max)
				{
					eps_max = eps_cur;
				}
				_A[i][j] = v_new;
			}
		S++;
		if ((eps_max < eps) || (S >= Nmax))
		{
			flag = true;
		}
		if (S <= print_time)
		{
			printf("\n Результат работы [ %i ] итерации метода простой итерации:\n", S);
			for (int i = _n; i >= 0; i--)
			{
				printf(" | ");
				for (int j = 0; j < _m + 1; j++)
					printf(" %.5lf ", _A[i][j]);
				printf(" |\n");
			}
		}
	}
}

inline void CalcOfError(long double **_A, long double **_DecisionSolution, const long double _a, const long double _b, const long double _c, const long double _d, const int _n, const int _m, long double &z)
{
	long double z_ = 0.L;
	z = 0;
	long double h = (_b - _a) / _n;
	long double k = (_d - _c) / _m;
	long double x;
	long double y;
#pragma omp parallel for firstprivate(x,y)
	for (register int i = 0; i < _n + 1; i++)
	{
		for (register int j = 0; j < _m + 1; j++)
		{
			x = _a + h*i;
			y = _c + k*j;
			_DecisionSolution[i][j] = MainTask(x, y);
		}
	}
	for (register int i = 1; i < _n; i++)
	{
		for (register int j = 1; j < _m; j++)
		{
			z_ = fabs(_DecisionSolution[i][j] - _A[i][j]);
			if (z_ > z)
			{
				z = z_;
			}
		}
	}
}



int main(int argv, char* argc[])
{
	setlocale(LC_ALL, "Rus");
	system("color 3F");
	system("title ЛР_3 - Задача Дирихле для уравнения Пуассона (МПИ)");
	//int Nmax = 10000; // максимальное число итерации
	int S = 0; // счётчик Итераций
	long double r;
	long double eps = 0.0000000001L; // мин. допустимый прирост
	long double eps_max = 0.L; // текущее значение
	long double eps_cur = 0.L; // для подсчёта текущего значения прироста
	long double a2, k2, h2; // ненулевые элементы матрицы (-А)

	int n = 4, m = 4; // размерность СЛАУ

	long double a, b, c, d; // границы области определения уравнения

	long double v_old; // старое значение преобразуемой компоненты вектора X
	long double v_new; // новое значение
	bool flag = false; // условие остановки

	long double start, end;
	int max, print_time;
	long double Z_sin = 0;

	printf("* * * * * * Ввод данных * * * * * *\n");
	printf(" Введите размерность сетки:\n");
	printf(" n = ");
	scanf("%i", &n);
	printf(" m = ");
	scanf("%i", &m);
	register long double **Matrix = new long double*[n + 1];
	double long **DecisionSolution = new long double*[n + 1];
	for (register int i = 0; i < m + 1; i++)
	{
		Matrix[i] = new long double[m + 1];
		DecisionSolution[i] = new long double[m + 1];
	}

	printf("\n Введите граничные значения:\n");
	printf(" a = ");
	scanf("%lf", &a);
	printf("\n b = ");
	scanf("%lf", &b);
	printf("\n c = ");
	scanf("%lf", &c);
	printf("\n d = ");
	scanf("%lf", &d);
	printf("\n Введите количество итераций:\n");
	printf("\n Nmax = ");
	scanf("%i", &max);
	printf("\n Сколько итераций процесса работы метода вывести на экран:\n");
	printf("\n = ");
	scanf("%i", &print_time);
	while (print_time > max)
	{
		printf("\n Количество итераций должно быть меньше максимальных, это же очевидно!\nВведите заного = ");
		scanf("%i", &print_time);
	}
	start = omp_get_wtime();
	StartValue(Matrix, n, m, a, b, c, d);
	system("cls");
	MethodZeidel(print_time, max, Matrix, a, b, c, d, n, m, S, eps_max, eps_cur, eps);
	long double** R = new long double*[n + 1];					//Правая часть СЛАУ
	for (register int i = 0; i < m + 1; i++)
	{
		R[i] = new long double[m + 1];
	}
	r = DiscrepancyOfSolution(Matrix, n, m, a, b, c, d,R);
	end = omp_get_wtime();
	CalcOfError(Matrix, DecisionSolution, a, b, c, d, n, m, Z_sin);

	printf("\n* * * * * * Справка задачи * * * * * *\n");
	printf(" Граничные значения:\n");
	printf(" a = [ %.2lf ] b = [ %.2lf ]\n", a, b);
	printf(" c = [ %.2lf ] d = [ %.2lf ]\n", c, d);
	printf(" Размерность сетки:\n");
	printf(" n = [ %i ] m = [ %i ]\n\n", n, m);
	printf("* * * * * * Справка метода * * * * * *\n");
	//printf(" Максимальное число итераций: [ %i ]\n", max);
	printf(" Минимально допустимый прирост: [ %.15lf ]\n", eps);
	printf(" Последнее максимальное значение прироста: [ %.15lf ]\n", eps_max);
	printf(" Последнее рабочее значение прироста: [ %.15lf ]\n", eps_cur);
	printf("* * * * * * * * * * * * * * * * * * * *\n\n");
	printf(" Результаты работы алгоритма МПИ:\n");
	printf(" Время работы алгоритма: [ %.2lf ] секунд\n", end - start);
	printf(" Процессоров: [ %i ]", omp_get_max_threads());
	printf("\n Матрица А =\n");

	for (int i = n; i >= 0; i--)
	{
		printf(" | ");
		for (int j = 0; j < m + 1; j++)
			printf(" %.3lf ", Matrix[i][j]);
		printf(" |\n");
	}
	printf("\n Матрица U (точного решения) =\n");
	for (int i = n; i >= 0; i--)
	{
		printf(" | ");
		for (int j = 0; j < m + 1; j++)
			printf(" %.3lf ", DecisionSolution[i][j]);
		printf(" |\n");
	}

	printf("\n Погрешность метода Z = [ %.15lf ]\n", Z_sin);
	printf("\n Невязка  r = [ %.15lf ]\n", r);
	printf(" Количество выполненных итераций: [ %i ]\n", S);
	system("pause");
	return 0;
}