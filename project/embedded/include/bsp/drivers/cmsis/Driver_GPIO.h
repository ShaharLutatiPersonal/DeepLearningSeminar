#ifndef DRIVER_GPIO_H_
#define DRIVER_GPIO_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "Driver_Common.h"

#define ARM_GPIO_API_VERSION ARM_DRIVER_VERSION_MAJOR_MINOR(1, 0) /* API version */

/****** GPIO Control Codes *****/

#define ARM_GPIO_PIN_CONTROL_Pos 0
#define ARM_GPIO_PIN_CONTROL_Msk (0xFFUL << ARM_GPIO_PIN_CONTROL_Pos)

/*----- GPIO Control Codes: Mode -----*/
#define ARM_GPIO_SET_PIN_MODE        (0x01UL << ARM_GPIO_PIN_CONTROL_Pos)
#define ARM_GPIO_SET_PIN_STRENGTH    (0x02UL << ARM_GPIO_PIN_CONTROL_Pos)
#define ARM_GPIO_SET_PIN_IRQ_TRIGGER (0x03UL << ARM_GPIO_PIN_CONTROL_Pos)

#define ARM_GPIO_PIN_MODE_Pos                  8
#define ARM_GPIO_PIN_MODE_Msk                  (7UL << ARM_GPIO_PIN_MODE_Pos)
#define _ARM_GPIO_PIN_MODE_DISABLED            (1UL)
#define _ARM_GPIO_PIN_MODE_INPUT               (2UL)
#define _ARM_GPIO_PIN_MODE_INPUT_PULL          (3UL)
#define _ARM_GPIO_PIN_MODE_PUSH_PULL           (4UL)
#define _ARM_GPIO_PIN_MODE_PUSH_PULL_ALTERNATE (5UL)
#define _ARM_GPIO_PIN_MODE_OPEN_DRAIN          (6UL)
#define ARM_GPIO_PIN_MODE_DISABLED             (_ARM_GPIO_PIN_MODE_DISABLED << ARM_GPIO_PIN_MODE_Pos)
#define ARM_GPIO_PIN_MODE_INPUT                (_ARM_GPIO_PIN_MODE_INPUT << ARM_GPIO_PIN_MODE_Pos)
#define ARM_GPIO_PIN_MODE_INPUT_PULL           (_ARM_GPIO_PIN_MODE_INPUT_PULL << ARM_GPIO_PIN_MODE_Pos)
#define ARM_GPIO_PIN_MODE_PUSH_PULL            (_ARM_GPIO_PIN_MODE_PUSH_PULL << ARM_GPIO_PIN_MODE_Pos)
#define ARM_GPIO_PIN_MODE_PUSH_PULL_ALTERNATE  (_ARM_GPIO_PIN_MODE_PUSH_PULL_ALTERNATE << ARM_GPIO_PIN_MODE_Pos)
#define ARM_GPIO_PIN_MODE_OPEN_DRAIN           (_ARM_GPIO_PIN_MODE_OPEN_DRAIN << ARM_GPIO_PIN_MODE_Pos)

#define ARM_GPIO_PIN_LEVEL_Pos   12
#define ARM_GPIO_PIN_LEVEL_Msk   (03UL << ARM_GPIO_PIN_LEVEL_Pos)
#define _ARM_GPIO_PIN_LEVEL_LOW  (01UL)
#define _ARM_GPIO_PIN_LEVEL_HIGH (02UL)
#define ARM_GPIO_PIN_LEVEL_LOW   (_ARM_GPIO_PIN_LEVEL_LOW << ARM_GPIO_PIN_LEVEL_Pos)
#define ARM_GPIO_PIN_LEVEL_HIGH  (_ARM_GPIO_PIN_LEVEL_HIGH << ARM_GPIO_PIN_LEVEL_Pos)

#define ARM_GPIO_IRQ_TRIGGER_Pos           14
#define ARM_GPIO_IRQ_TRIGGER_Msk           (7UL << ARM_GPIO_IRQ_TRIGGER_Pos)
#define _ARM_GPIO_IRQ_TRIGGER_EDGE_FALLING (1UL)
#define _ARM_GPIO_IRQ_TRIGGER_EDGE_RISING  (2UL)
#define _ARM_GPIO_IRQ_TRIGGER_EDGE_BOTH    (3UL)
#define _ARM_GPIO_IRQ_TRIGGER_LEVEL_LOW    (4UL)
#define _ARM_GPIO_IRQ_TRIGGER_LEVEL_HIGH   (5UL)
#define ARM_GPIO_IRQ_TRIGGER_EDGE_FALLING  (_ARM_GPIO_IRQ_TRIGGER_EDGE_FALLING << ARM_GPIO_IRQ_TRIGGER_Pos)
#define ARM_GPIO_IRQ_TRIGGER_EDGE_RISING   (_ARM_GPIO_IRQ_TRIGGER_EDGE_RISING << ARM_GPIO_IRQ_TRIGGER_Pos)
#define ARM_GPIO_IRQ_TRIGGER_EDGE_BOTH     (_ARM_GPIO_IRQ_TRIGGER_EDGE_BOTH << ARM_GPIO_IRQ_TRIGGER_Pos)
#define ARM_GPIO_IRQ_TRIGGER_LEVEL_LOW     (_ARM_GPIO_IRQ_TRIGGER_LEVEL_LOW << ARM_GPIO_IRQ_TRIGGER_Pos)
#define ARM_GPIO_IRQ_TRIGGER_LEVEL_HIGH    (_ARM_GPIO_IRQ_TRIGGER_LEVEL_HIGH << ARM_GPIO_IRQ_TRIGGER_Pos)

/****** GPIO specific error codes *****/
#define ARM_GPIO_ERROR_MODE         (ARM_DRIVER_ERROR_SPECIFIC - 1)  ///< Specified Mode not supported
#define ARM_GPIO_ERROR_IRQ_REASSIGN (ARM_DRIVER_ERROR_SPECIFIC - 2)  ///< Specified Mode not supported

/****** GPIO not connected definition ******/
#define ARM_GPIO_NOT_CONNECTED (0)
#define IS_GPIO_CONNECTED(gpio) (gpio != ARM_GPIO_NOT_CONNECTED)

// Function documentation
/**
  \fn          ARM_DRIVER_VERSION ARM_GPIO_GetVersion (void)
  \brief       Get driver version.
  \return      \ref ARM_DRIVER_VERSION

  \fn          ARM_GPIO_CAPABILITIES ARM_USART_GetCapabilities (void)
  \brief       Get driver capabilities
  \return      \ref ARM_GPIO_CAPABILITIES

  \fn          int32_t ARM_GPIO_Initialize (ARM_USART_SignalEvent_t cb_event)
  \brief       Initialize GPIO Interface.
  \param[in]   cb_event  Pointer to \ref ARM_GPIO_SignalEvent
  \return      \ref execution_status

  \fn          int32_t ARM_GPIO_Uninitialize (void)
  \brief       De-initialize GPIO Interface.
  \return      \ref execution_status

  \fn          int32_t ARM_GPIO_PowerControl (ARM_POWER_STATE state)
  \brief       Control GPIO Interface Power.
  \param[in]   state  Power state
  \return      \ref execution_status

  \fn          int32_t ARM_GPIO_Control (uint32_t control, uint32_t arg)
  \brief       Control GPIO Interface.
  \param[in]   control  Operation
  \param[in]   arg      Argument of operation (optional)
  \return      common \ref execution_status and driver specific \ref
  usart_execution_status

  \fn          ARM_GPIO_STATUS ARM_USART_GetStatus (void)
  \brief       Get GPIO status.
  \return      GPIO status \ref ARM_USART_STATUS

*/

typedef void (*gpio_irq_cb)(void);

typedef enum {
    LOW = 0,
    HIGH,
} ARM_GPIO_LEVEL;

typedef enum {
    CLOSE = 0,
    OPEN,
} ARM_GPIO_PIN_STATE;

typedef enum {
    GPIO_DRIVER_STATE_UNINITIALIZED,
    GPIO_DRIVER_STATE_INITIALIZED,
    GPIO_DRIVER_STATE_POWERED,
} ARM_GPIO_DRIVER_STATE;

/**
\brief GPIO Device Driver Capabilities.
*/
typedef struct _ARM_GPIO_CAPABILITIES {
    bool irqs;
    bool low_power;
} ARM_GPIO_CAPABILITIES;

/**
\brief Abstract structure of a GPIO pin.
*/
typedef struct _ARM_GPIO_PIN ARM_GPIO_PIN;

/**
\brief Access structure of the GPIO Driver.
*/
typedef struct _ARM_DRIVER_GPIO {
    ARM_DRIVER_VERSION(*GetVersion)
    (void);  ///< Pointer to \ref ARM_GPIO_GetVersion : Get driver version.
    ARM_GPIO_CAPABILITIES(*GetCapabilities)
    (void);                                          ///< Pointer to \ref ARM_GPIO_GetCapabilities : Get driver
                                                     ///< capabilities.
    int32_t (*Initialize)(void);                     ///< Pointer to \ref ARM_GPIO_Initialize :
                                                     ///< Initialize GPIO Interface.
    int32_t (*Uninitialize)(void);                   ///< Pointer to \ref ARM_GPIO_Uninitialize :
                                                     ///< De-initialize GPIO Interface.
    int32_t (*PowerControl)(ARM_POWER_STATE state);  ///< Pointer to \ref ARM_GPIO_PowerControl :
                                                     ///< Control GPIO Interface Power.
    int32_t (*Control)(ARM_GPIO_PIN *pin, uint32_t control,
                       uint32_t arg);  ///< Pointer to \ref ARM_GPIO_Control :
                                       ///< Control USART Interface.
    ARM_GPIO_LEVEL (*GetLevel)(ARM_GPIO_PIN *pin);
    int32_t (*SetLevel)(ARM_GPIO_PIN *pin, ARM_GPIO_LEVEL level);
} const ARM_DRIVER_GPIO;

extern ARM_DRIVER_GPIO Driver_GPIO;

#ifdef __cplusplus
}
#endif

#endif /* DRIVER_GPIO_H_ */
