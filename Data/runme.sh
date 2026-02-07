while [1]
do
	test=`sh -c 'echo $$; exec python get_stock_price.py& '`
	sleep(3600)
	kill $test
done


