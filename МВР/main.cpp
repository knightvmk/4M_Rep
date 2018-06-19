#include <stdlib.h>
#include <stdio.h>

#include <math.h>
#include <locale.h>

#include <Windows.h>
#include <omp.h>

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


inline void StartValue(long double **_A, const int _n, const int _m, const long double _a, const long double _b, const long double _c, const long double _d)
{
	long double h = 0.;
	long double k = 0.;
	long double x = 0.;
	long double y = 0.;
	h = (_b - _a) / _n;
	k = (_d - _c) / _m;
#pragma omp parallel for firstprivate(x, y)
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
inline void MethodZeidel(const int print_time, const int _max, long double **_A, const long double _a, const long double _b, const long double _c, const long double _d, const int _n, const int _m, int &S, long double &eps_max, long double &eps_cur, long double &eps , double &_w_opt)
{
	short Nmax = _max;
	register double **CopyMatrix = new double*[_n+1];
	for (int i = 0; i < _n+1; i++)
	{
		CopyMatrix[i] = new double[_m+1];
		for (int j = 0; j < _m+1; j++)
		{
			CopyMatrix[i][j] = _A[i][j];
		}
	}
	long double a2, k2, h2;
	long double v_old = 0.;
	long double v_new = 0.;
	bool flag = false;
	h2 = -pow((_n / (_b - _a)), 2);
	k2 = -pow((_m / (_d - _c)), 2);
	a2 = -2 * (h2 + k2);
	double f = 4;
	double w;
	double Lamda_Max;//используем для оценки
	//w = 2 - (_b - _a) / _n;//на нулевом шаге
	w = _w_opt;

	while (!flag)
	{
		eps_max = 0;
#pragma simd
		for (register int j = 1; j < _m; j++)
			for (register int i = 1; i < _n; i++)
			{
				v_old = _A[i][j];
				v_new = -w*(h2*(_A[i + 1][j] + _A[i - 1][j]) + k2*(_A[i][j + 1] + _A[i][j - 1]));
				v_new = v_new + (1-w)*a2*_A[i][j]+w*f;
				v_new = v_new / a2;
				eps_cur = fabs(v_old - v_new);
				if (eps_cur > eps_max)
				{
					eps_max = eps_cur;
				}
				CopyMatrix[i][j] = v_old;
				_A[i][j] = v_new;
			}
		S++;
		if ((eps_max < eps) || (S >= Nmax))
		{
			flag = true;
		}
		if (S <= print_time)
		{
			printf("\n Результат работы [ %i ] итерации метода Зейделя:\n", S);
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
double DiscrepancyOfSolution(long double **V, const int n, const int m, const long double a, const long double b, const long double c, const long double d)
{
	long double	a2, k2, h2;			//Ненулевые элементы матрицы
	long double  h, k;				//Шаги сетки
	register long double** F = new long double*[n + 1];					//Правая часть СЛАУ
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
#pragma omp critical
			if (r > rs)
				rs = r;
		}
	}

	return rs;
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
#pragma simd
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
	system("title ЛР_3 - Задача Дирихле для уравнения Пуассона (МВР)");
	//int Nmax = 10000; // максимальное число итерации
	int S = 0; // счётчик Итераций
	long double r;
	long double eps = 0.000001L; // мин. допустимый прирост
	long double eps_max = 0.L; // текущее значение
	long double eps_cur = 0.L; // для подсчёта текущего значения прироста
	long double a2, k2, h2; // ненулевые элементы матрицы (-А)
	double w;
	int n = 4, m = 4; // размерность СЛАУ

	long double a, b, c, d; // границы области определения уравнения

	long double v_old; // старое значение преобразуемой компоненты вектора X
	long double v_new; // новое значение
	double w_opt = 1.5;
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
	printf("\n Введите параметр w:\n");
	scanf("%lf", &w_opt);
	//printf("\n Введите оптимальный параметр w:\n");
	////printf("\n w = ");
	//scanf("%lf", &w);
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
	MethodZeidel(print_time, max, Matrix, a, b, c, d, n, m, S, eps_max, eps_cur, eps, w_opt);
	r = DiscrepancyOfSolution(Matrix, n, m, a, b, c, d);
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
	printf(" Результаты работы алгоритма МВР:\n");
	printf(" Время работы алгоритма: [ %.2lf ] секунд\n", end - start);
	printf(" Процессоров: [ %i ]\n", omp_get_max_threads());
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
	printf("\n Невязка по норме [inf] = [ %.15lf ]\n", r);
	printf(" Количество выполненных итераций: [ %i ]\n", S);
	system("pause");
	return 0;
}