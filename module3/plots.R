library(tidyverse)
setwd('~/github/cmput607-w18/module3/')

bin <- read_csv('bin_counts.csv', col_names=FALSE)
names(bin) <- c('Servo 1', 'Servo 2')
bin <- bin %>% gather(servo)

kan <- read_csv('kanerva_counts.csv', col_names=FALSE)
names(kan) <- c('Servo 1', 'Servo 2')
kan <- kan %>% gather(servo)

pdf('bin1.pdf')
bin %>%
  filter(value > 0) %>%
ggplot(aes(x=value, fill=servo)) +
  geom_histogram(position='identity', alpha=0.6) +
  scale_fill_discrete(name="Servo",labels=c("Servo 1","Servo 2")) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank(),
        legend.position = c(0.9, 0.8),
        legend.key=element_blank(),
        legend.title = element_blank(),
        axis.text=element_text(size=12)) +
  labs(x='Number of activations', y='Count')
dev.off()

pdf('kan1.pdf')
kan %>%
  filter(value > 0) %>%
ggplot(aes(x=value, fill=servo)) +
  geom_histogram(position='identity', alpha=0.6) +
  scale_fill_discrete(name="Servo",labels=c("Servo 1","Servo 2")) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank(),
        legend.position = c(0.9, 0.8),
        legend.key=element_blank(),
        legend.title = element_blank(),
        axis.text=element_text(size=12)) +
  labs(x='Number of activations', y='Count')
dev.off()

pos <- read_csv('Position.csv', col_names=FALSE)
names(pos) <- c('x', 'Servo 1', 'Servo 2')

pos <- gather(pos, servo, 'Position', -x)

pos_rup <- read_csv('Position RUPEEs.csv', 
		        col_names=c('x', 'r2', 'w2', 'rw2', 'r1', 'w1', 'rw1')) %>%
	    mutate(rupee1=r2+w2+rw2+r1+w1+rw1) %>%
	    select(x, rupee1)

ss_rup <- read_csv('Softswitch RUPEEs.csv', 
		        col_names=c('x', 'r2', 'w2', 'rw2', 'r1', 'w1', 'rw1')) %>%
	    mutate(rupee2=r2+w2+rw2+r1+w1+rw1) %>%
	    select(rupee2)

st_rup <- read_csv('Switchtime RUPEEs.csv', 
		        col_names=c('x', 'r2', 'w2', 'rw2', 'r1', 'w1', 'rw1')) %>%
	    mutate(rupee3=r2+w2+rw2+r1+w1+rw1) %>%
	    select(rupee3)

rup <- bind_cols(pos_rup, ss_rup, st_rup) %>%
	mutate(rupee=rupee1+rupee2+rupee3) %>%
	select(x, rupee)

pdf('rupee.pdf')
rup %>%
  ggplot(aes(x=x, y=rupee)) +
  geom_line() +
  geom_vline(xintercept=7000) + 
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank(),
        legend.title = element_blank(),
        legend.position = 'none') +
  labs(x='Time-step', y='RUPEE')
dev.off()
  
pos_ude <- read_csv('Position UDEs.csv', 
		        col_names=c('x', 'r2', 'w2', 'rw2', 'r1', 'w1', 'rw1')) %>%
	    mutate(ude1=r2+w2+rw2+r1+w1+rw1) %>%
	    select(x, ude1)

ss_ude <- read_csv('Softswitch UDEs.csv', 
		        col_names=c('x', 'r2', 'w2', 'rw2', 'r1', 'w1', 'rw1')) %>%
	    mutate(ude2=r2+w2+rw2+r1+w1+rw1) %>%
	    select(ude2)

st_ude <- read_csv('Switchtime UDEs.csv', 
		        col_names=c('x', 'r2', 'w2', 'rw2', 'r1', 'w1', 'rw1')) %>%
	    mutate(ude3=r2+w2+rw2+r1+w1+rw1) %>%
	    select(ude3)

ude <- bind_cols(pos_ude, ss_ude, st_ude) %>%
	mutate(udee=ude1+ude2+ude3) %>%
	select(x, udee)

pdf('ude.pdf')
ude %>%
  filter(x > 200) %>% 
  ggplot(aes(x=x, y=udee)) +
  geom_line() +
  geom_vline(xintercept=5000) +
  geom_vline(xintercept=5100) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank(),
        legend.title = element_blank(),
        legend.position = 'none') +
  labs(x='Time-step', y='UDE')
dev.off()

s1 <- bind_cols(filter(pos, servo=='Servo 1'),
			  pred %>%
			  filter(servo=='Servo 1') %>%
			  select(Prediction)) %>%
      select(-servo) %>%
      gather('p', 'value', -x)
  
s1 %>%
  filter(x > 19800) %>%
  ggplot(aes(x=x, y=value, color=p)) +
  geom_line() +
  geom_abline(slope=0, intercept=1) + 
  scale_fill_discrete(name="Servo",labels=c("Servo 1","Servo 2")) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank(),
        legend.key=element_blank(),
        legend.title = element_blank(),
        legend.position = 'none') +
  labs(x='Time-step', y='Position')
