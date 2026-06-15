# GO2014-ExRodarTodaSegundafeira3am
# Código de referência (Azure Functions - requer Azure Functions Core Tools, não roda no Colab):
#
# import azure.functions as func
# import logging
#
# app = func.FunctionApp()
#
# @app.schedule(schedule="0 0 3 * * 1", arg_name="myTimer", run_on_startup=False)
# def retrain_timer(myTimer: func.TimerRequest) -> None:
#     logging.info('Iniciando retreino automático...')
#     retrain_model()  # Função do GO2013-33PipelineDeRetreinoCompleto
#     logging.info('Retreino concluído!')

import datetime
import matplotlib.pyplot as plt


def proximas_execucoes(cron_dia_semana=0, cron_hora=3, n=8):
    """Calcula as próximas execuções de um cron 'toda segunda-feira às 3h' (0 0 3 * * 1)."""
    agora = datetime.datetime.now()
    base = agora.replace(hour=cron_hora, minute=0, second=0, microsecond=0)

    datas = []
    dia = base
    while len(datas) < n:
        if dia.weekday() == cron_dia_semana and dia > agora:
            datas.append(dia)
        dia += datetime.timedelta(days=1)
    return datas


if __name__ == "__main__":
    print("=== Azure Functions - Timer Trigger (demonstração local) ===")
    print()
    print("  Cron schedule: '0 0 3 * * 1' -> toda segunda-feira às 03:00")
    print()

    execucoes = proximas_execucoes()
    for d in execucoes:
        print(f"  -> Próxima execução: {d.strftime('%a %d/%m/%Y %H:%M')}")

    # Gráfico: linha do tempo das próximas execuções agendadas
    plt.figure(figsize=(10, 3))
    plt.eventplot([d.timestamp() for d in execucoes], orientation="horizontal", colors="tab:blue")
    plt.yticks([])
    plt.xticks(
        [d.timestamp() for d in execucoes],
        [d.strftime("%d/%m") for d in execucoes],
        rotation=45
    )
    plt.title("Próximas execuções do retreino automático (toda segunda, 03:00)")
    plt.tight_layout()
    plt.show()
