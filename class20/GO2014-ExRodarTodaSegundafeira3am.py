# GO2014-ExRodarTodaSegundafeira3am
import azure.functions as func
import logging


if __name__ == "__main__":
    app = func.FunctionApp()

    @app.schedule(schedule="0 0 3 * * 1", arg_name="myTimer", run_on_startup=False)
    def retrain_timer(myTimer: func.TimerRequest) -> None:
        logging.info('Iniciando retreino automático...')
        retrain_model()  # Função acima
        logging.info('Retreino concluído!')
