num_epochs = 100
clip_norm = 5



n_train = len(dl_train)//2
for epoc in range(num_epochs):
    t_start = time.time()
    cum_error = 0
    idx = 0
    si_snr_rel = 0
    shuffle(dl_train) # make sure that the batch average gradient is not affected by the same examples each epoch
    for inp1,inp2 in pairwise(dl_train):
        (x1,y1),(x2,y2) = load_dl(dl_train_type,inp1,inp2)
        x = torch.stack((x1,x2),0)
        y = torch.stack((y1,y2),0)
        optimizer.zero_grad()
        yy = y.to(device)
        xx = x.to(device)
        shat = model(yy.view(batch_size,1,-1))
        error = loss_by_relative_si_snr(xx,shat)
        error.backward()
        torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_norm)
        optimizer.step()
        cum_error += error.item()
        idx += 1
        precentile_done = round(100*(idx + 1)/n_train)
        progress_symbols = int(np.floor(precentile_done*80/100))
        print('\r['
                  + ('#')*progress_symbols
                  + (' ')*(80 - progress_symbols)
                  + ']' +
                  ' Epoch {}/{} progress {}/100%'.format(epoc + 1, num_epochs, precentile_done), end='')
    t_epoch = measure_epoch_time(t_start)
    print('\n')
    print('*'*33 + 'epoch  results' + '*'*33)
    print('epoch {} , si_snr relative {:+.2f} , epoch time {:+.2f} [min], estimated time last {:+.2f} [min]'.format(epoc,cum_error/idx,t_epoch/60,(num_epochs- epoc - 1)*(t_epoch/60)))
    print('*'*80)
    scheduler.step()
torch.save(model,'model_5.pth')
