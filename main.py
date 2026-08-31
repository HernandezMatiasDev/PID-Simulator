import ctypes
import math
import platform
from pathlib import Path
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent


# 1. Mapear el struct de C (basado en pid.h)
class PIDController(ctypes.Structure):
    _fields_ = [
        ("kp", ctypes.c_float),
        ("ki", ctypes.c_float),
        ("kd", ctypes.c_float),
        ("k_deriv_filter", ctypes.c_float),
        ("out_max", ctypes.c_float),
        ("out_min", ctypes.c_float),
        ("deadband", ctypes.c_int32),
        ("target", ctypes.c_int32),
        ("integral_sum", ctypes.c_float),
        ("previous_error", ctypes.c_int32),
        ("derivative", ctypes.c_float),
    ]


def _find_library() -> Path:
    """Busca la librería compilada al lado de este script, sea cual sea el SO."""
    system = platform.system()
    candidates = {
        "Windows": ["pid.dll"],
        "Darwin": ["pid.dylib", "pid.so"],
    }.get(system, ["pid.so"])

    for name in candidates:
        path = BASE_DIR / name
        if path.exists():
            return path

    raise FileNotFoundError(
        "No se encontro la libreria del PID (pid.dll / pid.so / pid.dylib) "
        f"junto a main.py. Compila pid.c primero (ver build.sh / build.bat). "
        f"Se busco en: {BASE_DIR}"
    )


# 2. Cargar la librería compartida (ruta absoluta, misma carpeta que main.py)
pid_lib = ctypes.CDLL(str(_find_library()))

# 3. Configurar firmas de las funciones
pid_lib.PID_Init.argtypes = [
    ctypes.POINTER(PIDController),
    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ctypes.c_float, ctypes.c_float, ctypes.c_int32,
]
pid_lib.PID_Compute.argtypes = [ctypes.POINTER(PIDController), ctypes.c_int32, ctypes.c_float]
pid_lib.PID_Compute.restype = ctypes.c_float
pid_lib.PID_SetTarget.argtypes = [ctypes.POINTER(PIDController), ctypes.c_int32]
pid_lib.PID_SetGains.argtypes = [ctypes.POINTER(PIDController), ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
pid_lib.PID_SetOutputLimits.argtypes = [ctypes.POINTER(PIDController), ctypes.c_float, ctypes.c_float, ctypes.c_int32]
pid_lib.PID_Reset.argtypes = [ctypes.POINTER(PIDController)]

# Instancia global del PID para el simulador
pid = PIDController()
pid_lib.PID_Init(ctypes.byref(pid), 1.0, 0.0, 0.5, 0.1, 100.0, -100.0, 1)  # Valores por defecto
pid_lib.PID_SetTarget(ctypes.byref(pid), 0)  # 0 = centro del riel (arriba positivo, abajo negativo)


def _safe(v):
    """JSON no admite NaN/Infinity: si algo se disparó, lo convertimos a un
    número finito en vez de mandar un JSON inválido que rompería el
    fetch().json() del frontend y congelaría la animación."""
    if isinstance(v, float) and not math.isfinite(v):
        return 0.0
    return v


def snapshot(output=None, dt=None):
    """Devuelve absolutamente todas las variables del struct + telemetria extra."""
    data = {
        "kp": _safe(pid.kp),
        "ki": _safe(pid.ki),
        "kd": _safe(pid.kd),
        "k_deriv_filter": _safe(pid.k_deriv_filter),
        "out_max": _safe(pid.out_max),
        "out_min": _safe(pid.out_min),
        "deadband": pid.deadband,
        "target": pid.target,
        "integral_sum": _safe(pid.integral_sum),
        # previous_error ya representa el error calculado en el ultimo Compute()
        "error": pid.previous_error,
        "derivative": _safe(pid.derivative),
    }
    if output is not None:
        data["output"] = _safe(output)
    if dt is not None:
        data["dt"] = dt
    return data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compute', methods=['POST'])
def compute():
    try:
        data = request.json
        dt = float(data['dt'])
        # Clampeamos dt para evitar picos (ej: pestaña en segundo plano) que
        # dispararian la derivada a valores absurdos.
        dt = max(0.0, min(dt, 0.1))
        output = pid_lib.PID_Compute(ctypes.byref(pid), int(data['position']), dt)
        if not math.isfinite(output):
            # Ante un valor invalido, no dejamos que el estado del PID quede
            # corrompido: lo reseteamos para que el sistema se recupere solo
            # en vez de quedar tildado mandando NaN para siempre.
            pid_lib.PID_Reset(ctypes.byref(pid))
            output = 0.0
        return jsonify(snapshot(output=output, dt=dt))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route('/update', methods=['POST'])
def update_params():
    try:
        data = request.json

        pid_lib.PID_SetTarget(ctypes.byref(pid), int(data['target']))
        pid_lib.PID_SetGains(
            ctypes.byref(pid),
            float(data['kp']), float(data['ki']), float(data['kd']),
            float(data.get('k_deriv_filter', pid.k_deriv_filter)),
        )
        pid_lib.PID_SetOutputLimits(
            ctypes.byref(pid),
            float(data.get('out_max', pid.out_max)),
            float(data.get('out_min', pid.out_min)),
            int(data.get('deadband', pid.deadband)),
        )

        if data.get('reset', False):
            pid_lib.PID_Reset(ctypes.byref(pid))

        return jsonify(snapshot())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == '__main__':
    # threaded=True evita que un pedido lento (p.ej. mientras se arrastra un
    # slider) bloquee el polling continuo de /compute con el servidor de
    # desarrollo de Flask, que por defecto es de un solo hilo.
    app.run(debug=True, threaded=True)