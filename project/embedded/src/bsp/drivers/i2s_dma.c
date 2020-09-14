/**********************************************************************************************************************/
/* Description: Silabs EFR32 UART driver implementation */
/**********************************************************************************************************************/

/**********************************************************************************************************************/
/* Includes                                                                                                           */
/**********************************************************************************************************************/

/* Own header */
#include <i2s_dma.h>

/* C Standards */
#include <assert.h>

/* Project dependencies */
#include <dma.h>

/* External dependencies */

/**********************************************************************************************************************/
/* Macros                                                                                                             */
/**********************************************************************************************************************/
#define ARM_SAI_DRIVER_VERSION ARM_DRIVER_VERSION_MAJOR_MINOR(1, 0) /* Driver version */

/**********************************************************************************************************************/
/* Variables */
/**********************************************************************************************************************/
static ARM_SAI_CAPABILITIES driver_capabilities = {
    .asynchronous       = 1,
    .synchronous        = 0,
    .protocol_user      = 0,
    .protocol_i2s       = 1,
    .protocol_justified = 1,
    .protocol_pcm       = 0,
    .protocol_ac97      = 0,
    .mono_mode          = 1,
    .companding         = 0,
    .mclk_pin           = 0,
    .event_frame_error  = 0,
    .reserved           = 0,
};

/**********************************************************************************************************************/
/* Private functions declaration                                                                                      */
/**********************************************************************************************************************/
static USART_I2sFormat_TypeDef translate_conf_databits(uint8_t databits);
static USART_Databits_TypeDef  translate_conf_format(uint8_t databits, uint8_t word_size);

/**********************************************************************************************************************/
/* API implementation                                                                                                 */
/**********************************************************************************************************************/

/**********************************************************************************************************************/
/* Description: Returns driver version */
/* Inputs     : none */
/* Outputs    : ARM_DRIVER_VERSION - driver version */
/**********************************************************************************************************************/
ARM_DRIVER_VERSION i2s_driver_get_version(void)
{
    return (ARM_DRIVER_VERSION){
        .api = ARM_SAI_API_VERSION,
        .drv = ARM_SAI_DRIVER_VERSION,
    };
}

/**********************************************************************************************************************/
/* Description: Returns driver capabilities */
/* Inputs     : none */
/* Outputs    : ARM_SAI_CAPABILITIES - driver capabilities */
/**********************************************************************************************************************/
ARM_SAI_CAPABILITIES i2s_driver_get_capabilities(void)
{
    return driver_capabilities;
}

/**********************************************************************************************************************/
/* Description:                                                                                                     */
/* Inputs     :  -                                                                                                */
/* Outputs    :  -                                                                                                 */
/* Notes      :                                                                                                     */
/**********************************************************************************************************************/
int32_t i2s_driver_initialize(i2s_resources_t *resources, ARM_SAI_SignalEvent_t cb_event)
{
    assert(resources != NULL);
    if (resources->_state == SAI_DRIVER_STATE_INITIALIZED)
    {
        return ARM_DRIVER_OK;
    }

    CMU_ClockEnable(cmuClock_HFPER, true);
    CMU_ClockEnable(resources->_clock, true);

    /* config peripheral */
    USART_InitI2s_TypeDef init_conf = USART_INITI2S_DEFAULT;
    init_conf.sync.enable           = usartDisable;
    init_conf.sync.baudrate         = resources->_conf->baudrate;
    init_conf.sync.databits         = translate_conf_databits(resources->_conf->databits);
    init_conf.sync.master           = resources->_conf->is_master;
    init_conf.format                = translate_conf_format(resources->_conf->databits, resources->_conf->word_size);
    init_conf.delay                 = true;
    init_conf.dmaSplit              = true;
    init_conf.justify               = resources->_conf->is_left_justified ? usartI2sJustifyLeft : usartI2sJustiryRight;
    init_conf.mono                  = resources->_conf->is_mono;

    /* config IOs */
    Driver_GPIO.Control(resources->_pins->i2s_rx, ARM_GPIO_SET_PIN_MODE, ARM_GPIO_PIN_MODE_INPUT);
    Driver_GPIO.Control(resources->_pins->i2s_tx, ARM_GPIO_SET_PIN_MODE, ARM_GPIO_PIN_MODE_PUSH_PULL);

    if (resources->_conf->is_master)
    {
        Driver_GPIO.Control(resources->_pins->i2s_clk, ARM_GPIO_SET_PIN_MODE, ARM_GPIO_PIN_MODE_PUSH_PULL);
        Driver_GPIO.Control(resources->_pins->i2s_ws, ARM_GPIO_SET_PIN_MODE, ARM_GPIO_PIN_MODE_PUSH_PULL);
    }
    else
    {
        return ARM_DRIVER_ERROR_UNSUPPORTED;
        //Driver_GPIO.Control(resources->_pins->i2s_clk, ARM_GPIO_SET_PIN_MODE, ARM_GPIO_PIN_MODE_INPUT);
        //Driver_GPIO.Control(resources->_pins->i2s_ws, ARM_GPIO_SET_PIN_MODE, ARM_GPIO_PIN_MODE_INPUT);
    }

    USART_InitAsync(resources->_peripheral, &init_conf);

    resources->_peripheral->ROUTEPEN
        |= (USART_ROUTEPEN_RXPEN | USART_ROUTEPEN_TXPEN | USART_ROUTEPEN_CLKPEN | USART_ROUTEPEN_CSPEN);
    resources->_peripheral->ROUTELOC0 |= resources->_pins->rx_loc;
    resources->_peripheral->ROUTELOC0 |= resources->_pins->tx_loc;
    resources->_peripheral->ROUTELOC0 |= resources->_pins->clk_loc;
    resources->_peripheral->ROUTELOC0 |= resources->_pins->ws_loc;

    resources->__status.tx_busy          = 0;
    resources->__status.rx_busy          = 0;
    resources->__status.tx_underflow     = 0;
    resources->__status.rx_overflow      = 0;
    resources->__status.rx_break         = 0;
    resources->__status.rx_framing_error = 0;
    resources->__status.rx_parity_error  = 0;

    resources->__callback = cb_event;

    resources->_state = SAI_DRIVER_STATE_INITIALIZED;

    return ARM_DRIVER_OK;
}

int32_t i2s_driver_uninitialize(i2s_resources_t *resources)
{
    assert(resources != NULL);
    if (resources->_state != SAI_DRIVER_STATE_INITIALIZED)
    {
        return ARM_DRIVER_ERROR;
    }

    resources->_state = SAI_DRIVER_STATE_UNINITIALIZED;
    Driver_GPIO.Control(resources->_pins->i2s_clk, ARM_GPIO_SET_PIN_MODE, ARM_GPIO_PIN_MODE_DISABLED);
    Driver_GPIO.Control(resources->_pins->i2s_ws, ARM_GPIO_SET_PIN_MODE, ARM_GPIO_PIN_MODE_DISABLED);
    Driver_GPIO.Control(resources->_pins->i2s_tx, ARM_GPIO_SET_PIN_MODE, ARM_GPIO_PIN_MODE_DISABLED);
    Driver_GPIO.Control(resources->_pins->i2s_rx, ARM_GPIO_SET_PIN_MODE, ARM_GPIO_PIN_MODE_DISABLED);

    return ARM_DRIVER_OK;
}

int32_t i2s_driver_power_control(i2s_resources_t *resources, ARM_POWER_STATE state)
{
    assert(resources != NULL);
    if (resources->_state == SAI_DRIVER_STATE_UNINITIALIZED)
    {
        return ARM_DRIVER_ERROR;
    }

    switch (state)
    {
        case ARM_POWER_OFF:
        {
            resources->_state = SAI_DRIVER_STATE_INITIALIZED;
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
            resources->_state = SAI_DRIVER_STATE_POWERED;
            break;
        }
        default:
        {
            break;
        }
    }

    return ARM_DRIVER_OK;
}

int32_t i2s_driver_send(i2s_resources_t *resources, const void *data, uint32_t num)
{
    return ARM_DRIVER_ERROR_UNSUPPORTED;
}

int32_t i2s_driver_receive(i2s_resources_t *resources, void *data, uint32_t num)
{
    assert(resources != NULL);
    if (resources->_state != SAI_DRIVER_STATE_POWERED)
    {
        return ARM_DRIVER_ERROR;
    }

    if (data == NULL)
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
//    USART_IntClear(resources->_peripheral, USART_IFC_RXFULL | USART_IFC_RXOF | USART_IFC_RXUF);
//    USART_IntEnable(resources->_peripheral, USART_IEN_RXDATAV | USART_IEN_RXFULL | USART_IEN_RXOF | USART_IEN_RXUF);
    LDMA_Transfer(...);
    return ARM_DRIVER_OK;
}

int32_t i2s_driver_transfer(i2s_resources_t *resources, const void *data_out, void *data_in, uint32_t num)
{
    return ARM_DRIVER_ERROR_UNSUPPORTED;
}

uint32_t i2s_driver_get_tx_count(i2s_resources_t *resources)
{
    return 0;
}

uint32_t i2s_driver_get_rx_count(i2s_resources_t *resources)
{
    return 0;
}

int32_t i2s_driver_control(i2s_resources_t *resources, uint32_t control, uint32_t arg)
{
    return ARM_DRIVER_ERROR_UNSUPPORTED;
}

ARM_SAI_STATUS i2s_driver_get_status(i2s_resources_t *resources)
{
    return resources->__status;
}

void i2s_driver_tx_irq_handler(i2s_resources_t *resources)
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

void i2s_driver_rx_irq_handler(i2s_resources_t *resources)
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

static USART_Databits_TypeDef translate_conf_databits(uint8_t databits)
{
    switch (databits)
    {
        case 8:
            return usartDatabits8;
        case 16:
            return usartDatabits16;
        default:
            return UINT8_MAX;
    }
}

static USART_I2sFormat_TypeDef translate_conf_format(uint8_t databits, uint8_t word_size)
{
    if (databits == 8)
    {
        switch (word_size)
        {
            case 8:
                return usartI2sFormatW8D8;
            case 16:
                return usartI2sFormatW16D8;
            case 32:
                return usartI2sFormatW32D8;
            default:
                return UINT8_MAX;
        }
    } else if (databits == 16)
    {
        switch (word_size)
        {
            case 16:
                return usartI2sFormatW16D16;
            case 32:
                return usartI2sFormatW32D16;
            default:
                return UINT8_MAX;
        }
    } else if (databits == 24)
    {
        switch (word_size)
        {
            case 32:
                return usartI2sFormatW32D24M;
            default:
                return UINT8_MAX;
        }
    } else if (databits == 32)
    {
        switch (word_size)
        {
            case 32:
                return usartI2sFormatW32D32;
            default:
                return UINT8_MAX;
        }
    }
    return UINT8_MAX;
}