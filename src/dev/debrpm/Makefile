# Run make deb to create a .deb package.
# P4A_ROOT must be defined. File tree under $P4A_ROOT/run will be captured.

LOCAL_INST=$(P4A_ROOT)/run
FINAL_PREFIX=/usr/local/p4a
TMP_BASE=/var/tmp/p4atmp
TMP_INST=$(TMP_BASE)$(FINAL_PREFIX)
LOCAL_ACCEL=$(P4A_ROOT)/src/p4a_accel
CHECKINSTALL=./checkinstall

PKG_NAME=par4all
PKG_GROUP=unknown
PKG_SOURCE=http://www.par4all.org
PKG_MAINT=

install-deb:
	@mkdir -p `dirname $(TMP_INST)`
	@cp -r $(LOCAL_INST) $(TMP_INST)
	@perl -i -pe 's,([ =])$(P4A_ROOT),\$1$(FINAL_PREFIX),' $(TMP_INST)/etc/par4all-rc.*
	perl -i -pe 's,(P4A_ROOT|P4A_DIST)([ =]+).+?\n,$$1$$2$(FINAL_PREFIX)\n,' $(TMP_INST)/etc/par4all-rc.*
	perl -i -pe 's,(P4A_ACCEL[ =]+).+?\n,$$1$(FINAL_PREFIX)/src/p4a_accel\n,' $(TMP_INST)/etc/par4all-rc.*
	@mkdir -p $(TMP_BASE)/etc/profile.d
	@ln -sf $(FINAL_PREFIX)/etc/par4all-rc.sh $(TMP_BASE)/etc/profile.d/
	@ln -sf $(FINAL_PREFIX)/etc/par4all-rc.csh $(TMP_BASE)/etc/profile.d/
	@mkdir -p $(TMP_INST)/src
	@cp -r $(LOCAL_ACCEL) $(TMP_INST)/src/

deb:
	@if [ -z "$(P4A_ROOT)" ]; then echo "ERROR: P4A_ROOT not defined"; exit 123; fi
	@rm -rf $(TMP_BASE)
	INSTALL_PREFIX=$(TMP_BASE) $(CHECKINSTALL) -D --install=no --nodoc --pkgname="$(PKG_NAME)" --pkggroup="$(PKG_GROUP)" --pkgsource="$(PKG_SOURCE)" --maintainer="$(PKG_MAINT)" make install-deb
	@rm -rf $(TMP_BASE)

