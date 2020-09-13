/**********************************************************************************************************************/
/* Description: main function */
/**********************************************************************************************************************/

/**********************************************************************************************************************/
/* Includes                                                                                                           */
/**********************************************************************************************************************/

/* C Standards */
#include <stdio.h>

/* Project dependencies */
#include <bsp_objects.h>
#include <em_chip.h>
#include <em_cmu.h>
#include <gpio.h>

/**********************************************************************************************************************/
/* Variables */
/**********************************************************************************************************************/
static volatile uint64_t tick         = 0;
volatile bool            recv_done    = false;
volatile bool            send_done    = false;
volatile bool            start_record = false;
volatile bool            start_send   = false;

void SysTick_Handler(void)
{
    tick++;
}

void volatile sleep(uint64_t ticks)
{
    __disable_irq();
    tick = 0;
    __enable_irq();
    while (tick < ticks)
        ;
}

void record_on(void)
{
    start_record = true;
}

void send_on(void)
{
    start_send = true;
}

void uart_cb(uint32_t event)
{
    if (event & ARM_USART_EVENT_SEND_COMPLETE)
    {
        send_done = true;
    }
    if (event & ARM_USART_EVENT_RECEIVE_COMPLETE)
    {
        __NOP();
    }
}
void sai_cb(uint32_t event)
{
    if (event & ARM_USART_EVENT_SEND_COMPLETE)
    {
        __NOP();
    }
    if (event & ARM_USART_EVENT_RECEIVE_COMPLETE)
    {
        recv_done = true;
    }
}

static char recv[32000];

int main(void)
{
    CHIP_Init();
    CMU_HFRCOFreqSet(cmuHFRCOFreq_38M0Hz);

    SysTick_Config(SystemCoreClock / 1000);
    Driver_GPIO.Initialize();
    Driver_GPIO.PowerControl(ARM_POWER_FULL);
    Driver_GPIO.Control(&GPIO_LED_PIN, ARM_GPIO_SET_PIN_MODE, ARM_GPIO_PIN_MODE_PUSH_PULL);
    Driver_GPIO.Control(&GPIO_MIC_ENABLE, ARM_GPIO_SET_PIN_MODE, ARM_GPIO_PIN_MODE_PUSH_PULL);
    Driver_GPIO.Control(&GPIO_BUTTON0_PIN, ARM_GPIO_SET_PIN_MODE, ARM_GPIO_PIN_MODE_INPUT);
    Driver_GPIO.Control(&GPIO_BUTTON1_PIN, ARM_GPIO_SET_PIN_MODE, ARM_GPIO_PIN_MODE_INPUT);

    gpio_irq_config_t start_record_conf = {
        .irq_trigger = ARM_GPIO_IRQ_TRIGGER_EDGE_FALLING,
        .callback    = record_on,
    };

    gpio_irq_config_t start_send_conf = {
        .irq_trigger = ARM_GPIO_IRQ_TRIGGER_EDGE_FALLING,
        .callback    = send_on,
    };

    Driver_GPIO.Control(&GPIO_BUTTON0_PIN, ARM_GPIO_SET_PIN_IRQ_TRIGGER, (uint32_t)&start_record_conf);
    Driver_GPIO.Control(&GPIO_BUTTON1_PIN, ARM_GPIO_SET_PIN_IRQ_TRIGGER, (uint32_t)&start_send_conf);

    Driver_UART_0.Initialize(uart_cb);
    Driver_UART_0.PowerControl(ARM_POWER_FULL);

    Driver_GPIO.SetLevel(&GPIO_MIC_ENABLE, HIGH);
    sleep(200);
    Driver_SAI_1.Initialize(sai_cb);
    Driver_SAI_1.PowerControl(ARM_POWER_FULL);
    sleep(200);

    while (true)
    {
        while (!start_record)
            ;
        start_record = false;
        Driver_SAI_1.Receive(recv, sizeof(recv));
        while (!recv_done)
            ;
        while (!start_send)
            ;
        recv_done  = false;
        start_send = false;
        Driver_UART_0.Send(recv, sizeof(recv));
        while (!send_done)
            ;
        send_done = false;
    }
    return 0;
}