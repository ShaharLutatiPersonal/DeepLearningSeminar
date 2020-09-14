################################################################################
#####                             Makefile                                 #####
################################################################################

### A general makefile for embedded project using arm-none-eabi-gcc
### Source file and include directories can be modified
### Jlink scripts for reset and erase a device are required

####################################### Project ################################
PROJECTNAME = Project
PROJECTTYPE = APP

####################################### Device #################################
FLASH_SIZE  = 0x100000
RAM_SIZE    = 0x40000
DEVICE      = EFR32MG12P332F1024GL125
JLINKDEVICE = EFR32MG12PxxxF1024

####################################### Toolchain ##############################
CC      = arm-none-eabi-gcc.exe
AS      = $(CC)
LD      = $(CC)
OBJCOPY = arm-none-eabi-objcopy.exe

####################################### Utils ##################################
PYTHON  := python
# ifeq ($(OS), Windows_NT)
# S       := \\
# MKDIR   := cmd /C mkdir /Q
# RMDIR   := cmd /C del /Q
# RM		:= cmd /C del /Q
# JLink   := JLink.exe
# else
S       := /
MKDIR   := mkdir -p
RMDIR   := rm -rf
RM 	    := rm -rf
JLink   := JLink
# endif

define to_lower
$(subst A,a,$(subst B,b,$(subst C,c,$(subst D,d,$(subst E,e,$(subst F,f,$(subst G,g,$(subst H,h,$(subst I,i,$(subst J,j,$(subst K,k,$(subst L,l,$(subst M,m,$(subst N,n,$(subst O,o,$(subst P,p,$(subst Q,q,$(subst R,r,$(subst S,s,$(subst T,t,$(subst U,u,$(subst V,v,$(subst W,w,$(subst X,x,$(subst Y,y,$(subst Z,z,$(1)))))))))))))))))))))))))))
endef

####################################### Config & Definitions ###################
VERBOSE := 0

ifeq ($(VERBOSE), 1)
vecho = @true
Q =
else
vecho = @echo
Q = @
endif

CONFIG       := DEBUG
LOWER_CONFIG := $(call to_lower,$(CONFIG))
OBJDIR		 := obj$(S)$(LOWER_CONFIG)
BINARYDIR    := bin$(S)$(LOWER_CONFIG)
ifeq ($(PROJECTTYPE), APP)
TARGET  := $(PROJECTNAME).elf
endif

####################################### Include Directories ####################
INCLUDE_DIRS = \
$(CURRENTDIR)include/bsp/drivers/cmsis \
$(CURRENTDIR)include/bsp/drivers/mcu \
$(CURRENTDIR)include/bsp/mcu/cmsis \
$(CURRENTDIR)include/bsp/mcu/device \
$(CURRENTDIR)include/bsp/mcu/emlib \
$(CURRENTDIR)include/bsp/objects \
$(CURRENTDIR)include/system \

####################################### C Source Files #########################
SRC_C = \
src/bsp/device/startup_efr32mg12p.c \
src/bsp/device/system_efr32mg12p.c \
src/bsp/drivers/gpio.c \
src/bsp/drivers/i2s_interrupt.c \
src/bsp/drivers/uart.c \
src/bsp/mcu/emlib/em_cmu.c \
src/bsp/mcu/emlib/em_core.c \
src/bsp/mcu/emlib/em_emu.c \
src/bsp/mcu/emlib/em_gpio.c \
src/bsp/mcu/emlib/em_usart.c \
src/bsp/objects/bsp_objects.c \
src/init/main.c \
src/system/global.c \


####################################### Asm Source Files #######################
SRC_S = \

SRC_s =

####################################### All source files #######################
SRC_FILES= $(SRC_C) $(SRC_S) $(SRc_s)

####################################### Libraries ##############################
EXTERNAL_LIBRARIES :=

####################################### Defines ################################
BUILD_DEFINES = $(DEVICE) ARM_MATH_CM4=1 flash_layout CMSIS_device_header=\"em_device.h\"

####################################### Flags ##################################
COMMON_FLAGS := -mcpu=cortex-m4 -mfloat-abi=soft -mfpu=fpv4-sp-d16 -mthumb -specs=nosys.specs -specs=nano.specs \
-ffunction-sections -fdata-sections
CFLAGS       := $(COMMON_FLAGS)
ASFLAGS      := $(COMMON_FLAGS)
LDFLAGS      := $(COMMON_FLAGS)
DEPFLAGS      = -MD -MF $(@:.o=.d)

####################################### Configuration dependent definitions ####
ifeq ($(CONFIG), DEBUG)
BUILD_DEFINES += DEBUG=1
CFLAGS += -O0
LDFLAGS +=
LINKERSCRIPT := linker_script/efr32mg12p.ld
else
BUILD_DEFINES += NDEBUG=1
CFLAGS += -Os
LDFLAGS += -flto
endif

#-fdata-sections -ffunction-sections
CFLAGS    += -std=c11 -ggdb \
$(addprefix -I,$(INCLUDE_DIRS)) $(addprefix -D,$(BUILD_DEFINES))

ASFLAGS += \
$(addprefix -I,$(INCLUDE_DIRS)) $(addprefix -D,$(BUILD_DEFINES))
#-Wl,--gc-sections
LDFLAGS   += -Wl,-Map=$(BINARYDIR)$(S)$(PROJECTNAME).map -Wl,--print-memory-usage -Wl,-gc-sections

####################################### Objects ################################
c_objects     = $(SRC_FILES:.c=.o)
c_s_objects   = $(c_objects:.s=.o)
c_s_S_objects = $(c_s_objects:.S=.o)

C_PATHS := $(sort $(dir $(SRC_FILES):.c))
s_PATHS := $(sort $(dir $(SRC_FILES):.s))
S_PATHS := $(sort $(dir $(SRC_FILES):.S))

vpath %.c $(C_PATHS)
vpath %.s $(s_PATHS)
vpath %.S $(S_PATHS)

ALLOBJS = $(addprefix $(subst $(S),/,$(OBJDIR)$(S)),$(notdir $(c_s_S_objects)))

####################################### Rules ##################################

all: $(BINARYDIR)$(S)$(PROJECTNAME).bin $(BINARYDIR)$(S)$(TARGET) | $(BINARYDIR)

$(BINARYDIR)$(S)$(PROJECTNAME).bin: $(BINARYDIR)$(S)$(TARGET)
	$(vecho) "Convert elf to binary"
	$(Q)$(OBJCOPY) -O binary $(BINARYDIR)$(S)$(TARGET) $(BINARYDIR)$(S)$(PROJECTNAME).bin

$(BINARYDIR)$(S)$(TARGET): $(ALLOBJS) | $(BINARYDIR)
	$(vecho) "Linking target"
	$(Q)$(LD) $^ $(LDFLAGS) -T$(LINKERSCRIPT) -o $@
# $(PYTHON) scripts/analyze_size.py $(BINARYDIR)$(S)$(TARGET) $(FLASH_SIZE) $(RAM_SIZE)

$(subst $(S),/,$(OBJDIR)$(S)%.o): %.c | $(OBJDIR)
	$(vecho) "Compiling file: $<"
	$(Q)$(CC) $(CFLAGS) -c $< -o $@ $(DEPFLAGS)
	
$(subst $(S),/,$(OBJDIR)$(S)%.o): %.s
	$(Q)$(AS) $(ASFLAGS) -s $< -o $@

$(subst $(S),/,$(OBJDIR)$(S)%.o): %.S
	$(Q)$(AS) $(ASFLAGS) -S $< -o $@

-include $(ALLOBJS:.o=.d)

$(OBJDIR):
	$(MKDIR) $(OBJDIR)

$(BINARYDIR):
	$(MKDIR) $(BINARYDIR)

clean:
	$(vecho) "Clean environment"
	$(RMDIR) $(OBJDIR) $(BINARYDIR)

reset:
	$(JLink) -device $(JLINKDEVICE) -if swd -speed auto -CommanderScript scripts/jlink/reset.jlink

erase:
	$(JLink) -device $(JLINKDEVICE) -if swd -speed auto -CommanderScript scripts/jlink/erase.jlink