/**********************************************************************************************************************/
/* Description: Silabs EFR32 UART driver implementation */
/**********************************************************************************************************************/

/**********************************************************************************************************************/
/* Includes                                                                                                           */
/**********************************************************************************************************************/

/* Own header */
#include <uart.h>

/* C Standards */
#include <assert.h>

/* Project dependencies */

/* External dependencies */

/**********************************************************************************************************************/
/* Macros                                                                                                             */
/**********************************************************************************************************************/
#define ARM_USART_DRIVER_VERSION ARM_DRIVER_VERSION_MAJOR_MINOR(1, 0) /* Driver version */

/**********************************************************************************************************************/
/* Variables */
/**********************************************************************************************************************/
static ARM_USART_CAPABILITIES driver_capabilities = {
    .asynchronous       = 1,
    .synchronous_master = 1,
    .synchronous_slave  = 0,
    .single_wire        = 0,
    .irda               = 0,
    .smart_card         = 0,
    .smart_card_clock   = 0,
    .flow_control_rts   = 0,
    .flow_control_cts   = 0,
    .event_tx_complete  = 1,
    .event_rx_timeout   = 0,
    .rts                = 0,
    .cts                = 0,
    .dtr                = 0,
    .dsr                = 0,
    .dcd                = 0,
    .ri                 = 0,
    .event_cts          = 0,
    .event_dsr          = 0,
    .event_dcd          = 0,
    .event_ri           = 0,
    .reserved           = 0,
};

/**********************************************************************************************************************/
/* Private functions declaration                                                                                      */
/**********************************************************************************************************************/

/**********************************************************************************************************************/
/* API implementation                                                                                                 */
/**********************************************************************************************************************/

/**********************************************************************************************************************/
/* Description: Returns driver version */
/* Inputs     : none */
/* Outputs    : ARM_DRIVER_VERSION - driver version */
/**********************************************************************************************************************/
ARM_DRIVER_VERSION uart_driver_get_version(void)
{
    return (ARM_DRIVER_VERSION){
        .api = ARM_USART_API_VERSION,
        .drv = ARM_USART_DRIVER_VERSION,
    };
}

/**********************************************************************************************************************/
/* Description: Returns driver capabilities */
/* Inputs     : none */
/* Outputs    : ARM_USART_CAPABILITIES - driver capabilities */
/**********************************************************************************************************************/
ARM_USART_CAPABILITIES uart_driver_get_capabilities(void)
{
    return driver_capabilities;
}

/**********************************************************************************************************************/
/* Description:                                                                                                     */
/* Inputs     :  -                                                                                                */
/* Outputs    :  -                                                                                                 */
/* Notes      :                                                                                                     */
/**********************************************************************************************************************/
int32_t uart_driver_initialize(uart_resources_t *resources, ARM_USART_SignalEvent_t cb_event)
{
    assert(resources != NULL);
    if (resources->_state == USART_DRIVER_STATE_INITIALIZED)
    {
        return ARM_DRIVER_OK;
    }

    CMU_ClockEnable(cmuClock_HFPER, true);
    CMU_ClockEnable(resources->_clock, true);

    /* config peripheral */
    USART_InitAsync_TypeDef init_conf = USART_INITASYNC_DEFAULT;
    init_conf.baudrate                = resources->_conf->baud_rate;
    init_conf.enable                  = usartDisable;

    /* config IOs */
    Driver_GPIO.Control(resources->_pins->uart_rx, ARM_GPIO_SET_PIN_MODE, ARM_GPIO_PIN_MODE_INPUT);
    Driver_GPIO.Control(resources->_pins->uart_tx, ARM_GPIO_SET_PIN_MODE, ARM_GPIO_PIN_MODE_PUSH_PULL);

    USART_InitAsync(resources->_peripheral, &init_conf);

    resources->_peripheral->ROUTEPEN |= (USART_ROUTEPEN_RXPEN | USART_ROUTEPEN_TXPEN);
    resources->_peripheral->ROUTELOC0 |= resources->_pins->rx_loc;
    resources->_peripheral->ROUTELOC0 |= resources->_pins->tx_loc;

    resources->__status.tx_busy          = 0;
    resources->__status.rx_busy          = 0;
    resources->__status.tx_underflow     = 0;
    resources->__status.rx_overflow      = 0;
    resources->__status.rx_break         = 0;
    resources->__status.rx_framing_error = 0;
    resources->__status.rx_parity_error  = 0;

    resources->__callback = cb_event;

    resources->_state = USART_DRIVER_STATE_INITIALIZED;

    return ARM_DRIVER_OK;
}

int32_t uart_driver_uninitialize(uart_resources_t *resources)
{
    assert(resources != NULL);
    if (resources->_state != USART_DRIVER_STATE_INITIALIZED)
    {
        return ARM_DRIVER_ERROR;
    }

    resources->_state = USART_DRIVER_STATE_UNINITIALIZED;
    Driver_GPIO.Control(resources->_pins->uart_tx, ARM_GPIO_SET_PIN_MODE, ARM_GPIO_PIN_MODE_DISABLED);
    Driver_GPIO.Control(resources->_pins->uart_rx, ARM_GPIO_SET_PIN_MODE, ARM_GPIO_PIN_MODE_DISABLED);

    return ARM_DRIVER_OK;
}

int32_t uart_driver_power_control(uart_resources_t *resources, ARM_POWER_STATE state)
{
    assert(resources != NULL);
    if (resources->_state == USART_DRIVER_STATE_UNINITIALIZED)
    {
        return ARM_DRIVER_ERROR;
    }

    switch (state)
    {
        case ARM_POWER_OFF:
        {
            resources->_state = USART_DRIVER_STATE_INITIALIZED;
            USART_Enable(resources->_peripheral, false);
            NVIC_DisableIRQ(resources->_tx_irq);
            NVIC_DisableIRQ(resources->_rx_irq);
            NVIC_ClearPendingIRQ(resources->_tx_irq);
            NVIC_ClearPendingIRQ(resources->_rx_irq);
            CMU_ClockEnable(resources->_clock, false);
            break;
        }
        case ARM_POWER_LOW:
        {
            return ARM_DRIVER_ERROR_UNSUPPORTED;
        }
        case ARM_POWER_FULL:
        {
            NVIC_ClearPendingIRQ(resources->_rx_irq);
            NVIC_ClearPendingIRQ(resources->_tx_irq);
            NVIC_EnableIRQ(resources->_rx_irq);
            NVIC_EnableIRQ(resources->_tx_irq);
            USART_Enable(resources->_peripheral, usartEnable);
            resources->_state = USART_DRIVER_STATE_POWERED;
            break;
        }
        default:
        {
            break;
        }
    }

    return ARM_DRIVER_OK;
}

int32_t uart_driver_send(uart_resources_t *resources, const void *data, uint32_t num)
{
    assert(resources != NULL);
    if (resources->_state != USART_DRIVER_STATE_POWERED)
    {
        return ARM_DRIVER_ERROR;
    }

    if (data == NULL || num == 0)
    {
        return ARM_DRIVER_ERROR_PARAMETER;
    }

    if (resources->__status.tx_busy)
    {
        return ARM_DRIVER_ERROR_BUSY;
    }

    resources->__status.tx_busy = true;

    resources->__send_buff  = data;
    resources->__send_len   = num;
    resources->__bytes_sent = 0;

    USART_IntClear(resources->_peripheral, USART_IFC_TXC | USART_IFC_TXOF | USART_IFC_TXUF);
    USART_IntEnable(resources->_peripheral, USART_IEN_TXC | USART_IEN_TXOF | USART_IEN_TXUF);
    USART_Tx(resources->_peripheral, resources->__send_buff[0]);

    return ARM_DRIVER_OK;
}

int32_t uart_driver_receive(uart_resources_t *resources, void *data, uint32_t num)
{
    assert(resources != NULL);
    if (resources->_state != USART_DRIVER_STATE_POWERED)
    {
        return ARM_DRIVER_ERROR;
    }

    if (data == NULL || num == 0)
    {
        return ARM_DRIVER_ERROR_PARAMETER;
    }

    if (resources->__status.rx_busy)
    {
        return ARM_DRIVER_ERROR_BUSY;
    }

    resources->__status.rx_busy = true;

    resources->__recv_buff  = data;
    resources->__recv_len   = num;
    resources->__bytes_recv = 0;

    resources->_peripheral->CMD = USART_CMD_CLEARRX;
    USART_IntClear(resources->_peripheral, USART_IFC_RXFULL | USART_IFC_RXOF | USART_IFC_RXUF);
    USART_IntEnable(resources->_peripheral, USART_IEN_RXDATAV | USART_IEN_RXFULL | USART_IEN_RXOF | USART_IEN_RXUF);

    return ARM_DRIVER_OK;
}

int32_t uart_driver_transfer(uart_resources_t *resources, const void *data_out, void *data_in, uint32_t num)
{
    return ARM_DRIVER_ERROR_UNSUPPORTED;
}

uint32_t uart_driver_get_tx_count(uart_resources_t *resources)
{
    return 0;
}

uint32_t uart_driver_get_rx_count(uart_resources_t *resources)
{
    return 0;
}

int32_t uart_driver_control(uart_resources_t *resources, uint32_t control, uint32_t arg)
{
    return ARM_DRIVER_ERROR_UNSUPPORTED;
}

ARM_USART_STATUS uart_driver_get_status(uart_resources_t *resources)
{
    return resources->__status;
}

int32_t uart_driver_set_modem_control(uart_resources_t *resources, ARM_USART_MODEM_CONTROL control)
{
    return ARM_DRIVER_ERROR_UNSUPPORTED;
}

ARM_USART_MODEM_STATUS uart_driver_get_modem_status(uart_resources_t *resources)
{
    return resources->__modem_status;
}

void uart_driver_tx_irq_handler(uart_resources_t *resources)
{
    uint32_t flags = USART_IntGetEnabled(resources->_peripheral);
    uint32_t event = 0;

    if (flags & USART_IF_TXC)
    {
        resources->__bytes_sent++;
        event |= ARM_USART_EVENT_TX_COMPLETE;

        if (resources->__bytes_sent == resources->__send_len)
        {
            USART_IntDisable(resources->_peripheral, USART_IEN_TXC | USART_IEN_TXOF | USART_IEN_TXUF);
            USART_IntClear(resources->_peripheral, USART_IFC_TXC | USART_IFC_TXOF | USART_IFC_TXUF);
            resources->__status.tx_busy = false;
            event |= ARM_USART_EVENT_SEND_COMPLETE;
        } else
        {
            USART_Tx(resources->_peripheral, resources->__send_buff[resources->__bytes_sent]);
        }
    }

    if (resources->__callback != NULL)
    {
        resources->__callback(event);
    }
}

void uart_driver_rx_irq_handler(uart_resources_t *resources)
{
    uint32_t flags = USART_IntGetEnabled(resources->_peripheral);
    uint32_t event = 0;

    if (flags & USART_IF_RXDATAV)
    {
        resources->__recv_buff[resources->__bytes_recv] = USART_Rx(resources->_peripheral);
        resources->__bytes_recv++;

        if (resources->__bytes_recv == resources->__recv_len)
        {
            USART_IntDisable(resources->_peripheral,
                             USART_IEN_RXDATAV | USART_IEN_RXFULL | USART_IEN_RXOF | USART_IEN_RXUF);
            resources->__status.rx_busy = false;
            event |= ARM_USART_EVENT_RECEIVE_COMPLETE;
        }
    }

    if (resources->__callback != NULL)
    {
        resources->__callback(event);
    }
}

/**********************************************************************************************************************/
/* Private functions implementation                                                                                   */
/**********************************************************************************************************************/