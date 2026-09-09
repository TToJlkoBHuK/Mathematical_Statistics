import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from scipy.optimize import linprog, minimize

np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')
np.set_printoptions(precision=6, suppress=True)

# ---------------------------------------------------------------- данные

A1_mid = np.array([[0.95, 1.00],
                   [1.05, 1.00],
                   [1.10, 1.00]])

R1_tomo = np.array([[1.0, 1.0],      # томография: неточны оба столбца
                    [1.0, 1.0],
                    [1.0, 1.0]])

R1_reg = np.array([[1.0, 0.0],       # регрессия: неточен только первый столбец
                   [1.0, 0.0],
                   [1.0, 0.0]])

A2_mid = np.array([[1.10, 0.90, 1.10],
                   [1.40, 1.00, 0.80],
                   [0.80, 1.40, 1.20]])

R2 = np.ones((3, 3))

LOG = []


def say(text=""):
    print(text)
    LOG.append(text)


def mat_str(M, w=10, prec=6):
    return "\n".join("   [" + "  ".join(f"{v:{w}.{prec}f}" for v in row) + "]" for row in M)


# ------------------------------------------- радиус особенности и матрица A'

def singularity_radius(Ac, R):
    # |Ac x| <= d*R|x| при x != 0  =>  d_min = min_x max_i |(Ac x)_i| / (R|x|)_i;
    # все строки R одинаковы, поэтому в каждом знаковом ортанте это ЛП
    m, n = Ac.shape
    r = R[0]
    best_t, best_x = np.inf, None

    for signs in product([1.0, -1.0], repeat=n):
        S = np.diag(signs)
        AS = Ac @ S                                  # x = S u, u >= 0
        c = np.zeros(n + 1)
        c[-1] = 1.0                                  # min t
        A_ub = np.vstack([np.hstack([AS, -np.ones((m, 1))]),
                          np.hstack([-AS, -np.ones((m, 1))])])
        b_ub = np.zeros(2 * m)
        A_eq = np.hstack([r.reshape(1, -1), [[0.0]]])   # (r, u) = 1
        b_eq = np.array([1.0])
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=[(0, None)] * (n + 1), method='highs')
        if res.success and res.fun < best_t:
            best_t = res.fun
            best_x = np.array(signs) * res.x[:n]

    return best_t, best_x


def build_singular_matrix(Ac, R, x):
    # E_ij = -(Ac x)_i * R_ij * sign(x_j) / (R|x|)_i  =>  (A_c + E) x = 0, |E| <= d_min*R
    E = -np.outer((Ac @ x) / (R @ np.abs(x)), np.sign(x)) * R
    return Ac + E, E


def check_membership(A_point, Ac, R, delta, tol=1e-9):
    dev = np.abs(A_point - Ac)
    lim = delta * R
    return bool(np.all(dev <= lim + tol)), dev, lim


def min_sigma_over_box(Ac, R, deltas, X):
    # min_A sigma_min(A) = min_||x||=1 sqrt(sum_i max(0, |Ac x|_i - d*(R|x|)_i)^2)
    P = np.abs(X @ Ac.T)
    Q = np.abs(X) @ R.T
    out = np.empty(len(deltas))
    for k, d in enumerate(deltas):
        vals = np.sqrt((np.maximum(P - d * Q, 0.0) ** 2).sum(axis=1))
        j = vals.argmin()

        def obj(v, d=d):                             # уточнение вокруг узла сетки
            v = v / np.linalg.norm(v)
            rr = np.maximum(np.abs(Ac @ v) - d * (R @ np.abs(v)), 0.0)
            return float(np.sqrt((rr ** 2).sum()))

        ref = minimize(obj, X[j], method='Nelder-Mead',
                       options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 4000})
        out[k] = min(vals[j], ref.fun)
    return out


def sphere_grid_2d(n=200001):
    th = np.linspace(0.0, np.pi, n)
    return np.column_stack([np.cos(th), np.sin(th)])


def sphere_grid_3d(n_theta=260, n_phi=520):
    th = np.linspace(0.0, np.pi, n_theta)
    ph = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing='ij')
    return np.column_stack([(np.sin(T) * np.cos(P)).ravel(),
                            (np.sin(T) * np.sin(P)).ravel(),
                            np.cos(T).ravel()])


# --------------------------------------------- 1. прямоугольная матрица A1

say("=" * 92)
say("1. ПРЯМОУГОЛЬНАЯ ИНТЕРВАЛЬНАЯ МАТРИЦА A1 (3x2)")
say("=" * 92)
say("mid A1 =")
say(mat_str(A1_mid, prec=4))
say("")

rect_data = {}

for name, R in (("Регрессия  rad = d*[1 0]", R1_reg),
                ("Томография rad = d*[1 1]", R1_tomo)):
    d_min, x_star = singularity_radius(A1_mid, R)
    A_sing, E = build_singular_matrix(A1_mid, R, x_star)
    inside, dev, lim = check_membership(A_sing, A1_mid, R, d_min)
    sv = np.linalg.svd(A_sing, compute_uv=False)
    minors = [np.linalg.det(A_sing[list(idx)]) for idx in [(0, 1), (0, 2), (1, 2)]]

    say(f"--- {name} ---")
    say(f"    Радиус особенности delta_min      = {d_min:.6f}")
    say(f"    Диапазон особенности              : delta >= {d_min:.6f}")
    say(f"    Вектор ядра x* (A' x* = 0)        = [{x_star[0]:.6f}, {x_star[1]:.6f}]")
    say("    Матрица A' (все миноры 2x2 = 0):")
    say(mat_str(A_sing))
    say(f"    A' принадлежит A1 (|A'-mid| <= d*R)= {'ДА' if inside else 'НЕТ'}"
        f"   max|A'-mid| = {dev.max():.6f}  <=  d*R_max = {lim.max():.6f}")
    say(f"    Сингулярные числа A'              = [{sv[0]:.6f}, {sv[1]:.3e}]")
    say(f"    rank(A') = {np.linalg.matrix_rank(A_sing, tol=1e-9)} (< 2)   "
        f"миноры 2x2 = [{minors[0]:.2e}, {minors[1]:.2e}, {minors[2]:.2e}]")
    say(f"    ||A' x*||_inf                     = {np.abs(A_sing @ x_star).max():.3e}")
    say("")

    rect_data[name] = (R, d_min, x_star, A_sing)

# проверка: одномерный перебор по c для x = (1, -c)
c_grid = np.linspace(0.5, 1.6, 1_100_001)
m = A1_mid[:, 0]
f_reg = np.max(np.abs(m[None, :] - c_grid[:, None]), axis=1)
f_tomo = f_reg / (1.0 + np.abs(c_grid))
say("--- Независимая проверка (прямой перебор по c: x = (1, -c)) ---")
say(f"    Регрессия : min_c max_i |m_i - c|          = {f_reg.min():.6f}"
    f"   при c = {c_grid[f_reg.argmin()]:.4f}")
say(f"    Томография: min_c max_i |m_i-c| / (1+|c|)  = {f_tomo.min():.6f}"
    f"   при c = {c_grid[f_tomo.argmin()]:.4f}  (= 1/27)")
say("")

SPLIT = len(LOG)

# ------------------------------------------------ 2. квадратная матрица A2

say("=" * 92)
say("2. КВАДРАТНАЯ ИНТЕРВАЛЬНАЯ МАТРИЦА A2 (3x3),  rad A2 = delta * ones(3,3)")
say("=" * 92)
say("mid A2 =")
say(mat_str(A2_mid, prec=4))
say(f"det(mid A2) = {np.linalg.det(A2_mid):.6f}   cond(mid A2) = {np.linalg.cond(A2_mid):.4f}")
say("")

d2_min, x2_star = singularity_radius(A2_mid, R2)
A2_sing, E2 = build_singular_matrix(A2_mid, R2, x2_star)
inside2, dev2, lim2 = check_membership(A2_sing, A2_mid, R2, d2_min)
sv2 = np.linalg.svd(A2_sing, compute_uv=False)

say(f"    Радиус особенности delta_min      = {d2_min:.6f}")
say(f"    Диапазон особенности              : delta >= {d2_min:.6f}")
say(f"    Вектор ядра x* (A2' x* = 0)       = [{x2_star[0]:.6f}, {x2_star[1]:.6f}, {x2_star[2]:.6f}]")
say("    Матрица A2':")
say(mat_str(A2_sing))
say(f"    A2' принадлежит A2                = {'ДА' if inside2 else 'НЕТ'}"
    f"   max|A2'-mid| = {dev2.max():.6f}  <=  delta = {lim2.max():.6f}")
say(f"    det(A2')                          = {np.linalg.det(A2_sing):.3e}")
say(f"    Сингулярные числа A2'             = [{sv2[0]:.6f}, {sv2[1]:.6f}, {sv2[2]:.3e}]")
say(f"    rank(A2') = {np.linalg.matrix_rank(A2_sing, tol=1e-9)} (< 3)   "
    f"||A2' x*||_inf = {np.abs(A2_sing @ x2_star).max():.3e}")
say("")

# проверка 1: формула Полякa-Рона  d_min = 1 / ||Ac^-1||_{inf,1}
Ainv = np.linalg.inv(A2_mid)
z_all = np.array(list(product([1.0, -1.0], repeat=3)))
norm_inf1 = np.max(np.abs(Ainv @ z_all.T).sum(axis=0))
say("--- Независимая проверка 1 (формула Полякa-Рона) ---")
say(f"    ||(mid A2)^-1||_{{inf,1}} = max_z ||A^-1 z||_1 = {norm_inf1:.6f}")
say(f"    delta_min = 1 / ||A^-1||_{{inf,1}}            = {1 / norm_inf1:.6f}"
    f"   (расхождение с ЛП: {abs(1 / norm_inf1 - d2_min):.2e})")
say("")

# проверка 2: det полилинеен => экстремумы в вершинах бруса; бисекция по delta
V = np.array(list(product([1.0, -1.0], repeat=9))).reshape(-1, 3, 3)


def det_range(delta):
    dets = np.linalg.det(A2_mid[None, :, :] + delta * V)
    return dets.min(), dets.max()


lo, hi = 0.0, 1.0
for _ in range(200):
    mid = 0.5 * (lo + hi)
    dmin_, dmax_ = det_range(mid)
    if dmin_ <= 0.0 <= dmax_:
        hi = mid
    else:
        lo = mid
say("--- Независимая проверка 2 (перебор 2^9 = 512 вершинных матриц) ---")
say("    det(A) полилинеен по элементам => экстремумы достигаются в вершинах бруса;")
say("    интервальная матрица особенна <=> 0 принадлежит [min det, max det].")
d_lo, d_hi = det_range(0.99 * d2_min)
say(f"    delta = 0.99*delta_min: [min det, max det] = [{d_lo:.6f}, {d_hi:.6f}]  -> 0 НЕ входит")
d_lo, d_hi = det_range(1.01 * d2_min)
say(f"    delta = 1.01*delta_min: [min det, max det] = [{d_lo:.6f}, {d_hi:.6f}]  -> 0 входит")
say(f"    Бисекция по delta даёт delta_min = {hi:.6f}   (расхождение с ЛП: {abs(hi - d2_min):.2e})")
say("")

# проверка 3: мера особенности mu(delta) = min sigma_min по всему брусу
say("--- Независимая проверка 3 (точный минимум sigma_min по интервальному брусу) ---")
say("    min_{A in A} sigma_min(A) = min_{||x||=1} sqrt( sum_i max(0, |Ac x|_i - d*(R|x|)_i)^2 )")
say("      матрица / случай     d = 0.90*d_min   d = 0.99*d_min       d = d_min")
X2chk, X3chk = sphere_grid_2d(), sphere_grid_3d(400, 800)
for label, Ac_, R_, dm_, Xg in (("A1, регрессия ", A1_mid, R1_reg, 0.075, X2chk),
                                ("A1, томография", A1_mid, R1_tomo, 1 / 27, X2chk),
                                ("A2, томография", A2_mid, R2, d2_min, X3chk)):
    v = min_sigma_over_box(Ac_, R_, [0.9 * dm_, 0.99 * dm_, dm_], Xg)
    say(f"      {label}         {v[0]:10.6f}       {v[1]:10.6f}        {v[2]:9.2e}")
say("      (> 0 => матрица регулярна;  = 0 => матрица особенна)")
say("")

# ------------------------------------------------------------ итоги

summary = pd.DataFrame([
    {'Матрица': 'A1 (3x2)', 'Случай': 'Регрессия', 'rad': 'd*[1,0;1,0;1,0]',
     'delta_min': f"{rect_data['Регрессия  rad = d*[1 0]'][1]:.6f}",
     'Диапазон особенности': f"delta >= {rect_data['Регрессия  rad = d*[1 0]'][1]:.6f}"},
    {'Матрица': 'A1 (3x2)', 'Случай': 'Томография', 'rad': 'd*ones(3,2)',
     'delta_min': f"{rect_data['Томография rad = d*[1 1]'][1]:.6f}",
     'Диапазон особенности': f"delta >= {rect_data['Томография rad = d*[1 1]'][1]:.6f}"},
    {'Матрица': 'A2 (3x3)', 'Случай': 'Томография', 'rad': 'd*ones(3,3)',
     'delta_min': f"{d2_min:.6f}",
     'Диапазон особенности': f"delta >= {d2_min:.6f}"},
]).set_index('Матрица')

say("=" * 92)
say("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
say("=" * 92)
for line in summary.to_string().split("\n"):
    say(line)
say("=" * 92)

# ------------------------------------------------------------ графики

def render_console(lines, filename):
    lines = "\n".join(lines).split("\n")
    width = 0.068 * max(len(s) for s in lines) + 0.25
    fig = plt.figure(figsize=(width, 0.1335 * len(lines) + 0.30), facecolor='#0f1b2d')
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off'); ax.set_facecolor('#0f1b2d')
    ax.text(0.006, 0.995, "\n".join(lines), family='DejaVu Sans Mono', fontsize=8.0,
            color='#e6edf3', va='top', ha='left', linespacing=1.35)
    fig.savefig(filename, dpi=200, facecolor='#0f1b2d', bbox_inches='tight')
    plt.close(fig)


render_console(LOG[:SPLIT], "Lab1_Console_A1.png")
render_console(LOG[SPLIT:], "Lab1_Console_A2.png")

X2 = sphere_grid_2d()
deltas = np.linspace(0, 0.12, 601)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Прямоугольная интервальная матрица $\\mathbf{A}_1\\in\\mathbb{IR}^{3\\times2}$: '
             'потеря столбцового ранга', fontsize=14)

d_reg = rect_data['Регрессия  rad = d*[1 0]'][1]
for i, (mi, col) in enumerate(zip(A1_mid[:, 0], ['#1f77b4', '#2ca02c', '#d62728'])):
    ax1.fill_between(deltas, mi - deltas, mi + deltas, color=col, alpha=0.18)
    ax1.plot(deltas, mi - deltas, color=col, lw=1.4)
    ax1.plot(deltas, mi + deltas, color=col, lw=1.4,
             label=f'$a_{{{i + 1}1}} = {mi:.2f} \\pm \\delta$')
ax1.axvline(d_reg, color='k', ls='--', lw=1.5)
ax1.plot([d_reg], [1.025], 'k*', ms=15, zorder=5)
ax1.annotate(f'$\\delta_{{\\min}} = {d_reg:.4f}$\n$a_{{i1}} = 1.025$',
             xy=(d_reg, 1.025), xytext=(d_reg + 0.012, 0.965),
             arrowprops=dict(arrowstyle='->', lw=1.2), fontsize=11)
ax1.set_xlabel('$\\delta$'); ax1.set_ylabel('значения элементов 1-го столбца')
ax1.set_title('Случай регрессии: первый столбец становится\nпропорционален второму')
ax1.legend(loc='lower left', fontsize=9)

for label, key, col in (('Регрессия  $\\mathrm{rad}=\\delta\\,[1\\;0]$', 'Регрессия  rad = d*[1 0]', '#1f77b4'),
                        ('Томография  $\\mathrm{rad}=\\delta\\,[1\\;1]$', 'Томография rad = d*[1 1]', '#d62728')):
    R, dm, _, _ = rect_data[key]
    curve = min_sigma_over_box(A1_mid, R, deltas, X2)
    ax2.plot(deltas, curve, color=col, lw=2, label=label)
    ax2.axvline(dm, color=col, ls='--', lw=1.2)
    ax2.annotate(f'$\\delta_{{\\min}}={dm:.4f}$', xy=(dm, 0),
                 xytext=(dm + 0.003, 0.045 + 0.02 * (col == '#1f77b4')), color=col, fontsize=11)
ax2.axhline(0, color='k', lw=1)
ax2.set_xlabel('$\\delta$')
ax2.set_ylabel('$\\min_{A\\in\\mathbf{A}_1}\\ \\sigma_{\\min}(A)$')
ax2.set_title('Мера особенности: наименьшее сингулярное число\nпо всему интервальному брусу')
ax2.legend(fontsize=10)
plt.tight_layout()
plt.savefig("Lab1_Rectangular.png", dpi=300, bbox_inches='tight')
plt.close(fig)

X3 = sphere_grid_3d()
deltas2 = np.linspace(0, 0.16, 401)
dets = np.array([det_range(d) for d in deltas2])
curve2 = min_sigma_over_box(A2_mid, R2, deltas2, X3)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Квадратная интервальная матрица $\\mathbf{A}_2\\in\\mathbb{IR}^{3\\times3}$: '
             'радиус особенности', fontsize=14)

ax1.fill_between(deltas2, dets[:, 0], dets[:, 1], color='#1f77b4', alpha=0.25,
                 label='диапазон $\\det A$, $A\\in\\mathbf{A}_2$')
ax1.plot(deltas2, dets[:, 0], color='#1f77b4', lw=1.6)
ax1.plot(deltas2, dets[:, 1], color='#1f77b4', lw=1.6)
ax1.axhline(0, color='k', lw=1.2)
ax1.axvline(d2_min, color='#d62728', ls='--', lw=1.6)
ax1.plot([d2_min], [0], 'r*', ms=14, zorder=5)
ax1.annotate(f'$\\delta_{{\\min}} = {d2_min:.6f}$', xy=(d2_min, 0),
             xytext=(d2_min + 0.014, -0.31), color='#d62728', fontsize=11,
             arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2))
ax1.set_xlabel('$\\delta$'); ax1.set_ylabel('$\\det A$')
ax1.set_title('Диапазон определителя по $2^9=512$ вершинным матрицам')
ax1.legend(fontsize=10)

ax2.plot(deltas2, curve2, color='#2ca02c', lw=2,
         label='$\\min_{A\\in\\mathbf{A}_2}\\sigma_{\\min}(A)$')
ax2.axhline(0, color='k', lw=1)
ax2.axvline(d2_min, color='#d62728', ls='--', lw=1.6, label=f'$\\delta_{{\\min}}={d2_min:.6f}$')
ax2.set_xlabel('$\\delta$'); ax2.set_ylabel('$\\min_{A\\in\\mathbf{A}_2}\\ \\sigma_{\\min}(A)$')
ax2.set_title('Мера особенности интервальной матрицы')
ax2.legend(fontsize=10)
plt.tight_layout()
plt.savefig("Lab1_Square.png", dpi=300, bbox_inches='tight')
plt.close(fig)

print("\nГрафики сохранены: Lab1_Console_A1.png, Lab1_Rectangular.png, "
      "Lab1_Console_A2.png, Lab1_Square.png")
